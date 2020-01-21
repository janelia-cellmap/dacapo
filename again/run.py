from again.config import Task, Model, Optimizer, Run
from again.models import create_model
from again.optimizers import create_optimizer
from again.prediction_types import Affinities
from again.report import MongoDbReport
from again.train import create_train_pipeline
import configargparse
import funlib.run
import gunpowder as gp

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


def run_local(run_config):

    print(
        f"Running task {run_config.task} "
        f"with mode {run_config.model}, "
        f"using optimizer {run_config.optimizer}")

    fmaps_in = run_config.task.data.channels
    fmaps_out = {
        Affinities: run_config.task.data.dims
    }[run_config.task.predict]

    model = create_model(run_config.model, fmaps_in, fmaps_out)
    loss = run_config.task.loss()
    optimizer = create_optimizer(run_config.optimizer, model)

    pipeline, request = create_train_pipeline(
        run_config.task,
        run_config.optimizer,
        model,
        loss,
        optimizer)

    print(f"Training model with {model.num_parameters()} parameters")
    print(f"Using data {run_config.task.data.filename}")

    report = MongoDbReport(options.mongo_db_host, 'again_v01', run_config)

    with gp.build(pipeline):
        for i in range(run_config.optimizer.num_iterations):

            batch = pipeline.request_batch(request)

            # TODO: add timing
            report.add_training_iteration(i, batch.loss, 1.0)

            # TODO:
            # periodic valiation
            # early stopping

    # TODO:
    # testing


def run_remote(run_config):

    funlib.run.run(
        command=f'python {__file__} '
                f'-t {run_config.task.config_file} '
                f'-m {run_config.model.config_file} '
                f'-o {run_config.optimizer.config_file}',
        num_cpus=2,
        num_gpus=1,
        queue='slowpoke',
        execute=True)


def run_all(run_configs, num_workers=1):

    print(f"Running {len(run_configs)} configs:")
    for run_config in run_configs[:10]:
        print(f"\t{run_config}")
    if len(run_configs) > 10:
        print(f"(and {len(run_configs) - 10} more...)")

    if num_workers > 1:

        from multiprocessing import Pool
        with Pool(num_workers) as pool:
            pool.map(run_remote, run_configs)

    else:

        for run_config in run_configs:
            run_local(run_config)


if __name__ == "__main__":

    options = parser.parse()
    run_config = Run(
        Task(options.task),
        Model(options.model),
        Optimizer(options.optimizer))
    run_local(run_config)
