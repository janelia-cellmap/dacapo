import dacapo

import click

import logging
from pathlib import Path
import sys

from . import click_config_file


@click.group()
def cli():
    # when calling a function through command line
    # the local directory doesn't seem to be added to
    # the path by default. Add it explicitly here.
    cwd = Path.cwd()
    sys.path.append(str(cwd.absolute()))


@cli.command()
@click.option("-t", "--tasks", required=True, type=click.Path(exists=True))
@click.option("-d", "--data", required=True, type=click.Path(exists=True))
@click.option("-m", "--models", required=True, type=click.Path(exists=True))
@click.option("-o", "--optimizers", required=True, type=click.Path(exists=True))
@click.option("-R", "--repetitions", required=True, type=int)
@click.option("-v", "--validation-interval", required=True, type=int)
@click.option("-s", "--snapshot-interval", required=True, type=int)
@click.option("-b", "--keep-best-validation", required=True, type=str)
@click.option("-n", "--num-workers", default=1, type=int)
@click_config_file.configuration_option(section="runs")
def run(
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

    logging.basicConfig(level=logging.INFO)

    task_configs = dacapo.config.find_task_configs(str(tasks))
    data_configs = dacapo.config.find_data_configs(str(data))
    model_configs = dacapo.config.find_model_configs(str(models))
    optimizer_configs = dacapo.config.find_optimizer_configs(str(optimizers))

    configs = dacapo.enumerate_runs(
        task_configs=task_configs,
        data_configs=data_configs,
        model_configs=model_configs,
        optimizer_configs=optimizer_configs,
        repetitions=repetitions,
        validation_interval=validation_interval,
        snapshot_interval=snapshot_interval,
        keep_best_validation=keep_best_validation,
    )

    dacapo.run_all(configs, num_workers=num_workers)


@cli.group()
def visualize():
    pass