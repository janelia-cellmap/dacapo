import attr
from tqdm import tqdm
import gunpowder as gp
import torch


from dacapo.store import MongoDbStore
from dacapo.validate import validate_remote
from dacapo.training_stats import TrainingStats
from dacapo.validation_scores import ValidationScores
from dacapo.batch_generators.default import DefaultBatchGenerator
from dacapo.gp import BatchSource


from pathlib import Path
import time
from typing import Optional


@attr.s
class RunExecution:
    repetitions: int = attr.ib()
    num_iterations: int = attr.ib()
    keep_best_validation: Optional[str] = attr.ib(default=None)
    num_workers: int = attr.ib(default=1)
    validation_interval: int = attr.ib(default=1000)
    snapshot_interval: int = attr.ib(default=0)
    bsub_flags: str = attr.ib(default="")
    batch: bool = attr.ib(default=True)

    def __attrs_post_init__(self):
        if self.keep_best_validation is not None:
            tokens = self.keep_best_validation.split(":")
            self.best_score_name = tokens[1]
            self.best_score_relation = {"min": min, "max": max}[tokens[0]]
        else:
            self.best_score_name = None


@attr.s
class Run:
    task: str = attr.ib()
    dataset: str = attr.ib()
    model: str = attr.ib()
    optimizer: str = attr.ib()
    name: str = attr.ib()
    execution_details: RunExecution = attr.ib()
    repetition: int = attr.ib()
    id: Optional[str] = attr.ib(default=None)

    def __attrs_post_init__(self):
        self.training_stats = TrainingStats()
        self.validation_scores = ValidationScores()
        self.started = None
        self.stopped = None
        self.num_parameters = None

    def start(self):
        if self.id is None:
            raise Exception("ID should have been initialized on upload to mongodb")

        # set torch flags:
        # TODO: make these configurable?
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        store = MongoDbStore()
        store.sync_run(self)

        batch_generator = DefaultBatchGenerator()

        if self.stopped is not None:
            return

        self.started = time.time()

        task = store.get_task(self.task)
        dataset = store.get_dataset(self.dataset)
        model = store.get_model(self.model)
        optimizer = store.get_optimizer(self.optimizer)

        # self.num_parameters = model.num_parameters + task.num_parameters
        self.outdir.mkdir(parents=True, exist_ok=True)

        checkpoint, starting_iteration = self.load_training_state(
            store, model, optimizer
        )
        if starting_iteration > 0:
            store.store_training_stats(self)

        store.sync_run(self)
        pipeline, request, predictor_keys = batch_generator.create_pipeline(
            task,
            dataset,
            model,
            optimizer,
            outdir=self.outdir,
            snapshot_every=self.execution_details.snapshot_interval,
        )

        # initialize model, heads, optimizer etc.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = model.instantiate(dataset).to(device)
        heads = [
            predictor.head(model, dataset).to(device) for predictor in task.predictors
        ]
        parameters = [
            {"params": backbone.parameters()},
        ]
        for head in heads:
            parameters.append(
                {"params": head.parameters()},
            )
        torch_optimizer = optimizer.instance(parameters)
        torch_losses = [loss.instantiate() for loss in task.losses]

        # load from checkpoint
        if checkpoint is not None:
            self._load_parameters(checkpoint, model, heads, torch_optimizer)

        with gp.build(pipeline):

            # main run loop
            for i in tqdm(
                range(starting_iteration, self.execution_details.num_iterations),
                desc="train",
            ):

                # TODO: Any exception thrown before the first batch is requested will properly
                # propogate back to the web front end, but after the first batch is requested
                # errors result in a freeze.
                batch = pipeline.request_batch(request)

                # train step
                self.train_step(
                    i,
                    device,
                    batch,
                    request,
                    backbone,
                    torch_optimizer,
                    task,
                    heads,
                    torch_losses,
                    predictor_keys,
                )

                # evaluation step
                if (
                    self.execution_details.validation_interval > 0
                    and i % self.execution_details.validation_interval == 0
                ):
                    self.save_validation_model(backbone, heads, i)
                    # async function. Spawn a new worker to run validation.
                    # if this model hapens to be the best: clean up past best
                    # and move the weights to the "best" checkpoint file
                    # TODO: async problems... what if validations finish out of order?
                    validate_remote(self, i)

                # sync with mongodb
                if i % 100 == 0 and i > 0:
                    store.store_training_stats(self)
                    self.save_training_state(backbone, heads, torch_optimizer, i)

        store.store_training_stats(self)
        self.stopped = time.time()
        store.sync_run(self)

    @property
    def outdir(self):
        path = Path("runs", self.id)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    def validation_outdir(self, iteration: int):
        path = Path("runs", self.id, "validations", str(iteration))
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def best_checkpoint(self):
        return Path(
            self.outdir,
            f"validation_best_{self.execution_details.best_score_name}.checkpoint",
        )

    def train_step(
        self,
        i,
        device,
        batch,
        request,
        backbone,
        torch_optimizer,
        task,
        heads,
        torch_losses,
        predictor_keys,
    ):
        snapshot_iteration = (
            self.execution_details.snapshot_interval > 0
            and i % self.execution_details.snapshot_interval == 0
        )

        if snapshot_iteration:
            self.save_snapshot(batch, request, f"snapshot_{i}.zarr")

        t1 = time.time()
        torch_optimizer.zero_grad()
        backbone_output = backbone.forward(
            torch.as_tensor(batch[gp.ArrayKeys.RAW].data, device=device)
        )
        losses = []
        outputs = []
        for predictor, head, loss in zip(task.predictors, heads, torch_losses):
            target_key, weight_key = predictor_keys[predictor.name]
            head_output = head.forward(backbone_output)
            outputs.append(head_output)
            loss_inputs = [
                head_output,
                torch.as_tensor(batch[target_key].data, device=device),
            ]
            if weight_key is not None:
                loss_inputs.append(
                    torch.as_tensor(batch[weight_key].data, device=device)
                )
            losses.append(loss.forward(*loss_inputs))
        total_loss = torch.prod(torch.stack(losses))
        total_loss.backward()
        torch_optimizer.step()
        t2 = time.time()

        # report training stats
        self.training_stats.add_training_iteration(
            i, total_loss.cpu().detach().numpy(), t2 - t1
        )

        # TODO: compare against upstream timing to see if there could be gains
        # with more cpus?
        # train_time = batch.profiling_stats.get_timing_summary(
        #     "Train", "process"
        # ).times[-1]

        if snapshot_iteration:
            output_spec = batch[gp.ArrayKeys.GT].spec.copy()
            output_spec.dtype = None
            extra_request = gp.BatchRequest()
            extra_batch = gp.Batch()
            backbone_key = gp.ArrayKey("BACKBONE_PREDICTION")
            extra_request[backbone_key] = output_spec.copy()
            extra_batch[backbone_key] = gp.Array(
                backbone_output.cpu().detach().numpy(),
                output_spec.copy(),
            )
            for head_output, predictor in zip(outputs, task.predictors):
                predictor_key = gp.ArrayKey(f"{predictor.name.upper()}_SOFT_PREDICTION")
                extra_batch[predictor_key] = gp.Array(
                    head_output.cpu().detach().numpy(),
                    output_spec.copy(),
                )
                extra_request[predictor_key] = output_spec.copy()
            self.save_snapshot(extra_batch, extra_request, f"snapshot_{i}.zarr")

    def save_snapshot(self, batch, request, snapshot_file_name):
        batch_size = None
        for key, array in batch.items():
            if batch_size is None:
                batch_size = array.data.shape[0]
            else:
                assert batch_size == array.data.shape[0]

        batches = [gp.Batch() for _ in range(batch_size)]
        for key, array in batch.items():
            for i in range(batch_size):
                batches[i][key] = gp.Array(array.data[i], array.spec)

        for i, batch in enumerate(batches):
            snapshot_pipeline = BatchSource(batch) + gp.ZarrWrite(
                dataset_names={
                    key: f"{i}/{str(key).lower()}" for key, _ in request.items()
                },
                output_dir=self.outdir / "snapshots",
                output_filename=snapshot_file_name,
            )
            with gp.build(snapshot_pipeline):
                snapshot_pipeline.request_batch(request)

    def get_saved_iterations(self):
        for f in self.outdir.iterdir():
            if f.name.endswith(".checkpoint"):
                tokens = f.name.split(".", maxsplit=1)
                try:
                    checkpoint_iteration = int(tokens[0])
                    yield checkpoint_iteration
                except ValueError:
                    pass

    def _load_parameters(self, filename, model, heads, optimizer=None):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for i, (name, head, loss) in enumerate(heads):
            head.load_state_dict(checkpoint[f"{name}_{i}_state_dict"])

    def save_training_state(self, backbone, heads, optimizer, iteration):
        """
        Keep the most recent model weights and the iteration they belong to.
        This allows a run to continue where it left off.
        """
        checkpoint_name = self.outdir / f"{iteration}.checkpoint"
        state_dicts = {
            "backbone_state_dict": backbone.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        for i, head in enumerate(heads):
            state_dicts[f"head_{i}_state_dict"] = head.state_dict()

        torch.save(
            state_dicts,
            checkpoint_name,
        )

        # cleanup and remove old iteration checkpoints
        for checkpoint_iteration in self.get_saved_iterations():
            if checkpoint_iteration < iteration:
                Path(self.outdir, f"{checkpoint_iteration}.checkpoint").unlink()

    def save_validation_model(self, backbone, heads, iteration):
        """
        Keep the most recent model weights and the iteration they belong to.
        This allows a run to continue where it left off.
        """
        backbone_checkpoint, head_checkpoints = self.get_validation_checkpoints(
            iteration
        )
        torch.save(
            {"model_state_dict": backbone.state_dict()},
            backbone_checkpoint,
        )

        # cleanup and remove old iteration checkpoints
        for head, checkpoint in zip(heads, head_checkpoints):
            torch.save(
                {"model_state_dict": head.state_dict()},
                checkpoint,
            )

    def get_validation_checkpoints(self, iteration):
        store = MongoDbStore()
        task = store.get_task(self.task)

        backbone_checkpoint = self.validation_outdir(iteration) / f"backbone.checkpoint"

        head_checkpoints = [
            self.validation_outdir(iteration) / f"{predictor.name}_head.checkpoint"
            for predictor in task.predictors
        ]
        return backbone_checkpoint, head_checkpoints

    def load_training_state(self, store, model, optimizer):
        """
        Load the most recent model weights and the iteration they belong to.
        Continue training from here.
        """
        checkpoint_iterations = list(self.get_saved_iterations())
        if len(checkpoint_iterations) == 0:
            return None, 0
        else:
            iteration = max(checkpoint_iterations)
            checkpoint = self.outdir / f"{iteration}.checkpoint"
            return checkpoint, iteration + 1

    def __repr__(self):
        return f"{self.id}"
