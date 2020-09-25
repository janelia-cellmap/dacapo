import dacapo

import click

import logging
from pathlib import Path
import sys

from . import click_config_file


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="WARNING",
)
def cli(log_level):
    # when calling a function through command line
    # the local directory doesn't seem to be added to
    # the path by default. Add it explicitly here.
    cwd = Path.cwd()
    sys.path.append(str(cwd.absolute()))

    logging.basicConfig(level=getattr(logging, log_level.upper()))


@cli.command()
@click.option(
    "-t", "--tasks", required=True, type=click.Path(exists=True, file_okay=False)
)
@click.option(
    "-d", "--data", required=True, type=click.Path(exists=True, file_okay=False)
)
@click.option(
    "-m", "--models", required=True, type=click.Path(exists=True, file_okay=False)
)
@click.option(
    "-o", "--optimizers", required=True, type=click.Path(exists=True, file_okay=False)
)
@click.option("-R", "--repetitions", required=True, type=int)
@click.option("-v", "--validation-interval", required=True, type=int)
@click.option("-s", "--snapshot-interval", required=True, type=int)
@click.option("-b", "--keep-best-validation", required=True, type=str)
@click.option("-n", "--num-workers", default=1, type=int)
@click_config_file.configuration_option(section="runs")
def run_all(
    tasks,
    data,
    models,
    optimizers,
    repetitions,
    validation_interval,
    snapshot_interval,
    keep_best_validation,
    num_workers,
):
    import dacapo.config

    task_configs = dacapo.config.find_task_configs(str(tasks))
    data_configs = dacapo.config.find_data_configs(str(data))
    model_configs = dacapo.config.find_model_configs(str(models))
    optimizer_configs = dacapo.config.find_optimizer_configs(str(optimizers))

    runs = dacapo.enumerate_runs(
        task_configs=task_configs,
        data_configs=data_configs,
        model_configs=model_configs,
        optimizer_configs=optimizer_configs,
        repetitions=repetitions,
        validation_interval=validation_interval,
        snapshot_interval=snapshot_interval,
        keep_best_validation=keep_best_validation,
    )

    dacapo.run_all(runs, num_workers=num_workers)


@cli.command()
@click.option(
    "-t", "--task", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "-d", "--data", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "-m", "--model", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "-o", "--optimizer", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option("-R", "--repetitions", required=True, type=int)
@click.option("-v", "--validation-interval", required=True, type=int)
@click.option("-s", "--snapshot-interval", required=True, type=int)
@click.option("-b", "--keep-best-validation", required=True, type=str)
@click.option("-n", "--num-workers", default=1, type=int)
def run_one(
    task,
    data,
    model,
    optimizer,
    repetitions,
    validation_interval,
    snapshot_interval,
    keep_best_validation,
    billing,
):

    task = dacapo.config.TaskConfig(task)
    data = dacapo.config.DataConfig(data)
    model = dacapo.config.ModelConfig(model)
    optimizer = dacapo.config.OptimizerConfig(optimizer)

    run = dacapo.Run(
        task,
        data,
        model,
        optimizer,
        int(repetitions),
        int(validation_interval),
        int(snapshot_interval),
        keep_best_validation,
    )
    dacapo.run_local(run)


@cli.group()
def visualize():
    pass