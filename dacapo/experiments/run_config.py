import attr
import tempfile
import hashlib
import numpy as np

from funlib.geometry import Coordinate
from funlib.persistence import prepare_ds, open_ds

from dacapo.store.array_store import LocalContainerIdentifier
from dacapo.store.converter import converter
from .architectures import ArchitectureConfig
from .datasplits import DataSplitConfig, DataSplit
from .tasks import TaskConfig, Task
from .trainers import TrainerConfig, Trainer, GunpowderTrainerConfig
from .starts import StartConfig
from .training_stats import TrainingStats
from .validation_scores import ValidationScores
from .model import Model

from bioimageio.core import test_model
from bioimageio.spec import save_bioimageio_package
from bioimageio.spec.model.v0_5 import (
    ModelDescr,
    WeightsDescr,
    PytorchStateDictWeightsDescr,
    Author,
    CiteEntry,
    LicenseId,
    HttpUrl,
    ArchitectureFromLibraryDescr,
    OutputTensorDescr,
    InputTensorDescr,
    BatchAxis,
    ChannelAxis,
    SpaceInputAxis,
    SpaceOutputAxis,
    Identifier,
    AxisId,
    TensorId,
    SizeReference,
    FileDescr,
    Doi,
    IntervalOrRatioDataDescr,
    ParameterizedSize,
    Version,
)

import torch
import zarr

from dacapo.tmp import np_to_funlib_array

from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@attr.s
class RunConfig:
    """
    A class to represent a configuration of a run that helps to structure all the tasks,
    architecture, training, and datasplit configurations.

    ...

    Attributes:
    -----------
    task_config: `TaskConfig`
        A config defining the Task to run that includes deciding the output of the model and
        different methods to achieve the goal.

    architecture_config: `ArchitectureConfig`
         A config that defines the backbone architecture of the model. It impacts the model's
         performance significantly.

    trainer_config: `TrainerConfig`
        Defines how batches are generated and passed for training the model along with defining
        configurations like batch size, learning rate, number of cpu workers and snapshot logging.

    datasplit_config: `DataSplitConfig`
        Configures the data available for the model during training or validation phases.

    name: str
        A unique name for this run to distinguish it.

    repetition: int
        The repetition number of this run.

    num_iterations: int
        The total number of iterations to train for during this run.

    validation_interval: int
        Specifies how often to perform validation during the run. It defaults to 1000.

    start_config : `Optional[StartConfig]`
        A starting point for continued training. It is optional and can be left out.
    """

    architecture_config: ArchitectureConfig = attr.ib(
        metadata={
            "help_text": "A config defining the Architecture to train. The architecture defines "
            "the backbone of your model. The majority of your models weights will be "
            "defined by the Architecture and will be very impactful on your models "
            "performance. There is no need to worry about the output since depending "
            "on the chosen task, additional layers will be appended to make sure "
            "the output conforms to the expected format."
        }
    )
    task_config: TaskConfig | None = attr.ib(
        default=None,
        metadata={
            "help_text": "A config defining the Task to run. The task defines the output "
            "of your model. Do you want semantic segmentations, instance segmentations, "
            "or something else? The task also lets you choose from different methods of "
            "achieving each of these goals."
        },
    )
    trainer_config: TrainerConfig | None = attr.ib(
        default=None,
        metadata={
            "help_text": "The trainer config defines everything related to how batches are generated "
            "and passed to the model for training. Things such as augmentations (adding noise, "
            "random rotations, transposing, etc.), batch size, learning rate, number of cpu_workers "
            "and snapshot logging will be configured here."
        },
    )
    datasplit_config: DataSplitConfig | None = attr.ib(
        default=None,
        metadata={
            "help_text": "The datasplit config defines what data will be available for your model during "
            "training or validation. Usually this involves simply reading data from a zarr, "
            "but if there is any preprocessing that needs to be done, that can be configured here."
        },
    )

    name: str | None = attr.ib(
        default=None,
        metadata={
            "help_text": "A unique name for this run. This will be saved so you and "
            "others can find this run. Keep it short and avoid special "
            "characters."
        },
    )

    repetition: int | None = attr.ib(
        default=None, metadata={"help_text": "The repetition number of this run."}
    )
    num_iterations: int | None = attr.ib(
        default=None, metadata={"help_text": "The number of iterations to train for."}
    )
    batch_size: int = attr.ib(
        default=1, metadata={"help_text": "The batch size for the training."}
    )
    num_workers: int = attr.ib(
        default=0,
        metadata={"help_text": "The number of workers for the DataLoader."},
    )

    validation_interval: int = attr.ib(
        default=1000, metadata={"help_text": "How often to perform validation."}
    )
    snapshot_interval: int | None = attr.ib(
        default=None, metadata={"help_text": "How often to save snapshots."}
    )

    start_config: Optional[StartConfig] = attr.ib(
        default=None, metadata={"help_text": "A starting point for continued training."}
    )

    learning_rate: float = attr.ib(
        default=1e-3,
        metadata={"help_text": "The learning rate for the optimizer (RAdam)."},
    )

    _device: torch.device | None = None
    _optimizer: torch.optim.Optimizer | None = None
    _lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    _model: torch.nn.Module | None = None
    _datasplit: DataSplitConfig | None = None
    _task: Task | None = None
    _trainer: Trainer | None = None
    _training_stats: TrainingStats | None = None
    _validation_scores: ValidationScores | None = None

    @property
    def train_until(self) -> int:
        return self.num_iterations

    @property
    def task(self) -> Task | None:
        if self._task is None and self.task_config is not None:
            self._task = self.task_config.task_type(self.task_config)
        return self._task

    @property
    def architecture(self) -> ArchitectureConfig:
        return self.architecture_config

    @property
    def trainer(self) -> TrainerConfig:
        return self.trainer_config

    @property
    def datasplit(self) -> DataSplit:
        if self._datasplit is None:
            self._datasplit = self.datasplit_config.datasplit_type(
                self.datasplit_config
            )
        return self._datasplit

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model(self) -> Model:
        if self._model is None:
            if self.task is not None:
                self._model = self.task.create_model(self.architecture)
            else:
                self._model = Model(self.architecture, torch.nn.Identity(), None)
            if self.start_config is not None:
                self.start_config.start_type(self.start_config).initialize_weights(
                    self._model, None
                )
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            self._optimizer = torch.optim.RAdam(
                lr=self.learning_rate,
                params=self.model.parameters(),
                decoupled_weight_decay=True,
            )
        return self._optimizer

    @property
    def lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        if self._lr_scheduler is None:
            self._lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.num_iterations // self.validation_interval,
                last_epoch=-1,
            )
        return self._lr_scheduler

    @property
    def training_stats(self):
        if self._training_stats is None:
            self._training_stats = TrainingStats()
        return self._training_stats

    @training_stats.setter
    def training_stats(self, value: TrainingStats):
        self._training_stats = value

    @property
    def validation_scores(self):
        if self._validation_scores is None and self.task is not None:
            self._validation_scores = ValidationScores(
                self.task.parameters,
                self.datasplit.validate,
                self.task.evaluation_scores,
            )
        return self._validation_scores

    @staticmethod
    def get_validation_scores(run_config) -> ValidationScores:
        """
        Static method to get the validation scores without initializing model, optimizer, trainer, etc.

        Args:
            run_config: The configuration for the run.
        Returns:
            The validation scores.
        Raises:
            AssertionError: If the task or datasplit types are not specified in the run_config.
        Examples:
            >>> validation_scores = Run.get_validation_scores(run_config)
            >>> validation_scores
            ValidationScores object

        """
        task_type = run_config.task_config.task_type
        datasplit_type = run_config.datasplit_config.datasplit_type

        task = task_type(run_config.task_config)
        datasplit = datasplit_type(run_config.datasplit_config)

        return ValidationScores(
            task.parameters, datasplit.validate, task.evaluation_scores
        )

    def __str__(self):
        return self.name

    def visualize_pipeline(self, bind_address="0.0.0.0", bind_port=0):
        """
        Visualizes the pipeline for the run, including all produced arrays.

        Args:
            bind_address : str
                Bind address for Neuroglancer webserver
            bind_port : int
                Bind port for Neuroglancer webserver

        Examples:
            >>> run.visualize_pipeline()

        """
        if not isinstance(self.trainer, GunpowderTrainerConfig):
            raise NotImplementedError(
                "Only GunpowderTrainerConfig is supported for visualization"
            )
        if not hasattr(self.trainer, "_pipeline"):
            from ..store.create_store import create_array_store

            array_store = create_array_store()
            self.trainer.build_batch_provider(
                self.datasplit.train,
                self.model,
                self.task,
                array_store.snapshot_container(self.name),
            )
        self.trainer.visualize_pipeline(bind_address, bind_port)

    def save_bioimage_io_model(
        self,
        path: Path,
        authors: list[Author],
        cite: list[CiteEntry] | None = None,
        license: str = "MIT",
        input_test_image_path: Path | None = None,
        output_test_image_path: Path | None = None,
        checkpoint: int | str | None = None,
        in_voxel_size: Coordinate | None = None,
        test_saved_model: bool = False,
    ):
        # TODO: Fix this import. Importing here due to circular imports.
        # The weights store takes a Run to figure out where weights are saved,
        # but Run needs to import the weights store to load the weights.
        from dacapo.store.create_store import create_weights_store

        weights_store = create_weights_store()
        if checkpoint == "latest":
            checkpoint = weights_store.latest_iteration(self)
        if checkpoint is not None:
            weights_store.load_weights(self, checkpoint)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            input_axes = [
                BatchAxis(),
                ChannelAxis(
                    channel_names=[
                        Identifier(f"in_c{i}")
                        for i in range(self.architecture.num_in_channels)
                    ]
                ),
            ]
            input_shape = self.architecture.input_shape
            in_voxel_size = (
                np.array(in_voxel_size)
                if in_voxel_size is not None
                else np.array((1,) * input_shape.dims)
            )

            input_axes += [
                SpaceInputAxis(
                    id=AxisId(f"d{i}"),
                    size=ParameterizedSize(min=s, step=s),
                    scale=scale,
                )
                for i, (s, scale) in enumerate(zip(input_shape, in_voxel_size))
            ]
            data_descr = IntervalOrRatioDataDescr(type="float32")

            if input_test_image_path is None:
                input_test_image_path = tmp / "input_test_image.npy"
                test_image = np.random.random(
                    (
                        1,
                        self.architecture.num_in_channels,
                        *self.architecture.input_shape,
                    )
                ).astype(np.float32)
                np.save(input_test_image_path, test_image)

            input_descr = InputTensorDescr(
                id=TensorId("raw"),
                axes=input_axes,
                test_tensor=FileDescr(source=str(input_test_image_path)),
                data=data_descr,
            )

            output_shape = self.model.compute_output_shape(input_shape)[1]
            out_voxel_size = self.model.scale(in_voxel_size)
            context_units = Coordinate(
                np.array(input_shape) * in_voxel_size
            ) - Coordinate(np.array(output_shape) * out_voxel_size)
            context_out_voxels = Coordinate(np.array(context_units) / out_voxel_size)

            output_axes = [
                BatchAxis(),
                ChannelAxis(
                    channel_names=(
                        self.task.channels
                        if self.task is not None
                        else [
                            f"c{i}" for i in range(self.architecture.num_out_channels)
                        ]
                    )
                ),
            ]
            output_axes += [
                SpaceOutputAxis(
                    id=AxisId(f"d{i}"),
                    size=SizeReference(
                        tensor_id=TensorId("raw"),
                        axis_id=AxisId(f"d{i}"),
                        offset=-c,
                    ),
                    scale=s,
                )
                for i, (c, s) in enumerate(zip(context_out_voxels, out_voxel_size))
            ]
            if output_test_image_path is None:
                output_test_image_path = tmp / "output_test_image.npy"
                test_out_image = (
                    self.model.eval()(torch.from_numpy(test_image).float())
                    .detach()
                    .numpy()
                )
                np.save(output_test_image_path, test_out_image)
            output_descr = OutputTensorDescr(
                id=TensorId(
                    self.task.__class__.__name__.lower().replace("task", "")
                    if self.task is not None
                    else self.architecture_config.name
                ),
                axes=output_axes,
                test_tensor=FileDescr(source=str(output_test_image_path)),
            )

            pytorch_architecture = ArchitectureFromLibraryDescr(
                callable="from_yaml",
                kwargs={"config_yaml": converter.unstructure(self)},
                import_from="dacapo.experiments.run_config",
            )

            weights_path = tmp / "model.pth"
            torch.save(self.model.state_dict(), weights_path)
            with open(weights_path, "rb", buffering=0) as f:
                weights_hash = hashlib.file_digest(f, "sha256").hexdigest()

            my_model_descr = ModelDescr(
                name=self.name,
                description="A model trained with DaCapo",
                authors=authors,
                cite=[
                    CiteEntry(
                        text="paper",
                        doi=Doi("10.1234something"),
                    )
                ],
                license=LicenseId(license),
                documentation=HttpUrl(
                    "https://github.com/janelia-cellmap/dacapo/blob/main/README.md"
                ),
                git_repo=HttpUrl("https://github.com/janelia-cellmap/dacapo"),
                inputs=[input_descr],
                outputs=[output_descr],
                weights=WeightsDescr(
                    pytorch_state_dict=PytorchStateDictWeightsDescr(
                        source=weights_path,
                        sha256=weights_hash,
                        architecture=pytorch_architecture,
                        pytorch_version=Version(torch.__version__),
                    ),
                ),
            )

            if test_saved_model:
                summary = test_model(my_model_descr)
                summary.display()

            logger.info(
                "package path:",
                save_bioimageio_package(my_model_descr, output_path=path),
            )

    def data_loader(self) -> torch.utils.data.DataLoader:
        dataset = self.trainer.iterable_dataset(
            self.datasplit.train,
            self.model.input_shape,
            self.model.output_shape,
            self.task.predictor,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def to(self, device: torch.device):
        self._device = device
        self._model = self.model.to(self.device)
        # TODO: investigate a simple `.to(self.device)` alternative
        self.move_optimizer(self.device)

    def move_optimizer(
        self, device: torch.device, empty_cuda_cache: bool = False
    ) -> None:
        """
        Moves the optimizer to the specified device.

        Args:
            device: The device to move the optimizer to.
            empty_cuda_cache: Whether to empty the CUDA cache after moving the optimizer.
        Raises:
            AssertionError: If the optimizer state is not a dictionary.
        Examples:
            >>> run.move_optimizer(device)
            >>> run.optimizer
            Optimizer object

        """
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        if empty_cuda_cache:
            torch.cuda.empty_cache()

    def resume_training(self, stats_store, weights_store) -> int:
        # Parse existing stats
        self.training_stats = stats_store.retrieve_training_stats(self.name)
        self.validation_scores.scores = (
            stats_store.retrieve_validation_iteration_scores(self.name)
        )

        # how far have we trained and validated?
        trained_until = self.training_stats.trained_until()
        validated_until = self.validation_scores.validated_until()

        # remove validation past existing training stats
        if validated_until > trained_until:
            logger.info(
                f"Trained until {trained_until}, but validated until {validated_until}! "
                "Deleting extra validation stats"
            )
            self.validation_scores.delete_after(trained_until)

        # logger.info current training state
        logger.info(f"Current state: trained {trained_until}/{self.num_iterations}")

        # read weights of the latest iteration
        latest_weights_iteration = weights_store.latest_iteration(self)
        latest_weights_iteration = (
            0 if latest_weights_iteration is None else latest_weights_iteration
        )
        weights = None

        # check if existing weights are consistent with existing training stats
        if trained_until > 0:
            # no weights are stored, training stats are inconsistent, delete them
            if latest_weights_iteration is None:
                logger.warning(
                    f"Run {self.name} was previously trained until {trained_until}, but no weights are "
                    "stored. Will restart training from scratch."
                )

                trained_until = 0
                self.training_stats.delete_after(0)
                self.validation_scores.delete_after(0)

            # weights are stored, but not enough so some stats are inconsistent, delete the inconsistent ones
            elif latest_weights_iteration < trained_until:
                logger.warning(
                    f"Run {self.name} was previously trained until {trained_until}, but the latest "
                    f"weights are stored for iteration {latest_weights_iteration}. Will resume training "
                    f"from {latest_weights_iteration}."
                )

                trained_until = latest_weights_iteration
                self.training_stats.delete_after(trained_until)
                self.validation_scores.delete_after(trained_until)
                weights = weights_store.retrieve_weights(
                    self, iteration=trained_until
                )

            # perfectly in sync. We can continue training
            elif latest_weights_iteration == trained_until:
                logger.info(f"Resuming training from iteration {trained_until}")

                weights = weights_store.retrieve_weights(
                    self, iteration=trained_until
                )

            # weights are stored past the stored training stats, log this inconsistency
            # but keep training
            elif latest_weights_iteration > trained_until:
                weights = weights_store.retrieve_weights(
                    self, iteration=latest_weights_iteration
                )
                logger.error(
                    f"Found weights for iteration {latest_weights_iteration}, but "
                    f"self {self.name} was only trained until {trained_until}. "
                )

            # load the weights that we want to resume training from
            if weights is not None:
                self.model.load_state_dict(weights.model)
                self.optimizer.load_state_dict(weights.optimizer)
        return trained_until

    def train_step(self, raw: torch.Tensor, target: torch.Tensor, weight: torch.Tensor):
        self.optimizer.zero_grad()

        predicted = self.model.forward(raw.float().to(self.device))

        predicted.retain_grad()
        loss = self.task.loss.compute(
            predicted,
            target.float().to(self.device),
            weight.float().to(self.device),
        )
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item(), {
            "prediction": predicted.detach(),
            "gradients": predicted.grad.detach(),
        }

    def save_snapshot(
        self,
        iteration: int,
        batch: dict[str, torch.Tensor],
        batch_out: dict[str, torch.Tensor],
        snapshot_container: LocalContainerIdentifier,
    ):
        snapshot_zarr = zarr.open(snapshot_container.container, "a")
        # remove batch dim from all snapshot arrays

        raw = batch["raw"]
        gt = batch["target"]
        target = batch["target"]
        weight = batch["weight"]
        mask = batch["mask"]
        prediction = batch_out["prediction"]
        gradients = batch_out["gradients"]

        in_voxel_size = self.datasplit.train[0].raw.voxel_size
        out_voxel_size = self.datasplit.train[0].gt.voxel_size
        ndims = in_voxel_size.dims
        input_shape = Coordinate(raw.shape[-ndims:])
        output_shape = Coordinate(gt.shape[-ndims:])
        context = (input_shape * in_voxel_size - output_shape * out_voxel_size) / 2
        in_shift = context * 0
        out_shift = context

        (raw,) = [
            np_to_funlib_array(in_array.cpu().numpy(), in_shift, in_voxel_size)
            for in_array in [raw]
        ]
        (gt, target, weight, mask, prediction, gradients) = [
            np_to_funlib_array(out_array.cpu().numpy(), out_shift, out_voxel_size)
            for out_array in [gt, target, weight, mask, prediction, gradients]
        ]

        snapshot_arrays = {
            "volumes/raw": raw,
            "volumes/gt": gt,
            "volumes/target": target,
            "volumes/weight": weight,
            "volumes/mask": mask,
            "volumes/prediction": prediction,
            "volumes/gradients": gradients,
        }
        for k, v in snapshot_arrays.items():
            snapshot_array_identifier = snapshot_container.array_identifier(k)
            array_name = f"{snapshot_array_identifier.container}/{snapshot_array_identifier.dataset}"
            if k not in snapshot_zarr:
                array = prepare_ds(
                    array_name,
                    shape=(0, *v.shape),
                    offset=v.roi.offset,
                    voxel_size=v.voxel_size,
                    axis_names=("iteration^", *v.axis_names),
                    dtype=v.dtype if v.dtype != bool else np.uint8,
                    mode="w",
                )
            else:
                array = open_ds(array_name, mode="r+")

            # neuroglancer doesn't allow bools
            if not v.dtype == bool:
                data = v[:]
            else:
                data = v[:].astype(np.uint8) * 255

            # add an extra dimension so that the shapes match
            array._source_data.append(data[None, :])
            iterations = array.attrs.setdefault("iterations", list())
            iterations.append(iteration)


def from_yaml(config_yaml: dict) -> torch.nn.Module:
    run_config: RunConfig = converter.structure(config_yaml, RunConfig)
    return run_config.model
