from again.config import TaskConfig, ModelConfig, OptimizerConfig
from again.models import create_model
from again.optimizers import create_optimizer
from again.store import MongoDbStore
from again.train import create_train_pipeline
from again.training_stats import TrainingStats
from again.validate import validate
from again.validation_scores import ValidationScores
from tqdm import tqdm
import configargparse
import funlib.run
import gunpowder as gp
import hashlib
import os
import time


parser = configargparse.ArgParser(
    default_config_files=['~/.config/again', './again.conf'])
parser.add(
    '-c', '--config',
    is_config_file=True,
    help="The config file to use.")
parser.add(
    '-t', '--task',
    help="The task to run.")
parser.add(
    '-m', '--model',
    help="The model to use.")
parser.add(
    '-o', '--optimizer',
    help="The optimizer to use.")
parser.add(
    '-r', '--repetition',
    help="Which repetition to run.")


class Run:

    def __init__(
            self,
            task_config,
            model_config,
            optimizer_config,
            repetition):

        self.task_config = task_config
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.repetition = repetition

        self.training_stats = TrainingStats()
        self.validation_scores = ValidationScores()
        self.started = None
        self.stopped = None

        run_hash = hashlib.md5()
        run_hash.update(self.task_config.id.encode())
        run_hash.update(self.model_config.id.encode())
        run_hash.update(self.optimizer_config.id.encode())
        run_hash.update(str(self.repetition).encode())
        self.id = run_hash.hexdigest()

    def start(self):

        store = MongoDbStore()
        store.sync_run(self)

        if self.stopped is not None:
            print(f"SKIP: Run {self} was already completed earlier.")
            return

        self.started = time.time()

        model = create_model(
            self.task_config,
            self.model_config)
        loss = self.task_config.loss()
        optimizer = create_optimizer(self.optimizer_config, model)

        outdir = os.path.join('runs', self.id)
        os.makedirs(outdir, exist_ok=True)

        self.model_config.num_parameters = model.num_parameters()
        store.sync_run(self)
        pipeline, request = create_train_pipeline(
            self.task_config,
            self.model_config,
            self.optimizer_config,
            model,
            loss,
            optimizer,
            outdir=outdir,
            snapshot_every=100)

        validation_interval = 500

        with gp.build(pipeline):

            for i in tqdm(
                    range(self.optimizer_config.num_iterations),
                    desc="train"):

                batch = pipeline.request_batch(request)

                train_time = batch.profiling_stats.get_timing_summary(
                    'Train',
                    'process').times[-1]
                self.training_stats.add_training_iteration(
                    i,
                    batch.loss,
                    train_time)

                if i % validation_interval == 0 and i > 0:
                    scores = validate(
                        self.task_config,
                        self.model_config,
                        model,
                        store_results=os.path.join(outdir, 'validate.zarr'))
                    self.validation_scores.add_validation_iteration(
                        i,
                        scores)
                    store.store_validation_scores(self)

                if i % 100 == 0 and i > 0:
                    store.store_training_stats(self)

        store.store_training_stats(self)
        self.stopped = time.time()
        store.sync_run(self)

        # TODO:
        # testing

    def to_dict(self):

        return {
            'task_config': self.task_config.id,
            'model_config': self.model_config.id,
            'optimizer_config': self.optimizer_config.id,
            'repetition': self.repetition,
            'started': self.started,
            'stopped': self.stopped,
            'id': self.id
        }

    def __repr__(self):
        return f"{self.task_config} with {self.model_config}, " \
            f"using {self.optimizer_config}, repetition {self.repetition}"


def enumerate_runs(
        task_configs,
        model_configs,
        optimizer_configs,
        repetitions=1):

    runs = []
    for task_config in task_configs:
        for model_config in model_configs:
            for optimizer_config in optimizer_configs:
                for repetition in range(repetitions):
                    runs.append(Run(
                        task_config,
                        model_config,
                        optimizer_config,
                        repetition))
    return runs


def run_local(run):

    print(
        f"Running task {run.task_config} "
        f"with mode {run.model_config}, "
        f"using optimizer {run.optimizer_config}")

    run.start()


def run_remote(run):

    funlib.run.run(
        command=f'python {__file__} '
                f'-t {run.task_config.config_file} '
                f'-m {run.model_config.config_file} '
                f'-o {run.optimizer_config.config_file} '
                f'-r {run.repetition}',
        num_cpus=2,
        num_gpus=1,
        queue='slowpoke',
        execute=True)


def run_all(runs, num_workers=1):

    print(f"Running {len(runs)} configs:")
    for run in runs[:10]:
        print(f"\t{run}")
    if len(runs) > 10:
        print(f"(and {len(runs) - 10} more...)")

    if num_workers > 1:

        from multiprocessing import Pool
        with Pool(num_workers) as pool:
            pool.map(run_remote, runs)

    else:

        for run in runs:
            run_local(run)


if __name__ == "__main__":

    options = parser.parse_known_args()[0]
    run = Run(
        TaskConfig(options.task),
        ModelConfig(options.model),
        OptimizerConfig(options.optimizer),
        int(options.repetition))
    run_local(run)
