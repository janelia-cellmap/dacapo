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
    "-t",
    "--tasks",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="The directory of task configs.",
)
@click.option(
    "-d",
    "--data",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="The directory of data configs.",
)
@click.option(
    "-m",
    "--models",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="The directory of model configs.",
)
@click.option(
    "-o",
    "--optimizers",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="The directory of optimizer configs.",
)
@click.option(
    "-R",
    "--repetitions",
    required=True,
    type=int,
    help="Number of times to repeat each combination of (task, data, model, optimizer).",
)
@click.option(
    "-v",
    "--validation-interval",
    required=True,
    type=int,
    help="How many iterations between each validation run.",
)
@click.option(
    "-s",
    "--snapshot-interval",
    required=True,
    type=int,
    help="How many iterations between each saved snapshot.",
)
@click.option(
    "-b",
    "--keep-best-validation",
    required=True,
    type=str,
    help="Definition of what is considered the 'best' validation",
)
@click.option(
    "-n",
    "--num-workers",
    default=1,
    type=int,
    help="How many workers to spawn on to run jobs in parallel.",
)
@click.option(
    "-bf", "--bsub-flags", default=None, type=str, help="flags to pass to bsub"
)
@click.option(
    "--batch",
    default=False,
    type=bool,
    help="Whether to run the jobs as interactive or not.",
)
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
    bsub_flags,
    batch,
):
    import dacapo.config

    task_configs = dacapo.config.find_task_configs(str(tasks))
    data_configs = dacapo.config.find_data_configs(str(data))
    model_configs = dacapo.config.find_model_configs(str(models))
    optimizer_configs = dacapo.config.find_optimizer_configs(str(optimizers))

    if num_workers > 1:
        assert any(["-P" in flag for flag in bsub_flags]), "billing must be provided"

    runs = dacapo.enumerate_runs(
        task_configs=task_configs,
        data_configs=data_configs,
        model_configs=model_configs,
        optimizer_configs=optimizer_configs,
        repetitions=repetitions,
        validation_interval=validation_interval,
        snapshot_interval=snapshot_interval,
        keep_best_validation=keep_best_validation,
        bsub_flags=bsub_flags,
        batch=batch,
    )

    dacapo.run_all(runs, num_workers=num_workers)


@cli.command()
@click.option(
    "-t",
    "--task",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="The task to run.",
)
@click.option(
    "-d",
    "--data",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="The data to run.",
)
@click.option(
    "-m",
    "--model",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="The model to run.",
)
@click.option(
    "-o",
    "--optimizer",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="The optimizer to run.",
)
@click.option(
    "-R", "--repetitions", required=True, type=int, help="The repitition to run"
)
@click.option(
    "-v",
    "--validation-interval",
    required=True,
    type=int,
    help="The number of training iterations between validation iterations.",
)
@click.option(
    "-s",
    "--snapshot-interval",
    required=True,
    type=int,
    help="The number of training iterations between each snapshot.",
)
@click.option(
    "-b",
    "--keep-best-validation",
    required=True,
    type=str,
    help="Defines what the 'best' iteration is.",
)
def run_one(
    task,
    data,
    model,
    optimizer,
    repetitions,
    validation_interval,
    snapshot_interval,
    keep_best_validation,
):

    import dacapo.config

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


@cli.command()
@click.option(
    "-n",
    "--name",
    required=True,
    type=str,
    help="The name of the run whose weights you want to use.",
)
@click.option(
    "-pd",
    "--predict-data",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="The config file of the data you want to predict.",
)
@click.option(
    "-dc",
    "--daisy-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="The config file for daisy."
)
@click.option(
    "-t",
    "--tasks",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="The directory of task configs.",
)
@click.option(
    "-d",
    "--data",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="The directory of data configs.",
)
@click.option(
    "-m",
    "--models",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="The directory of model configs.",
)
@click.option(
    "-o",
    "--optimizers",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="The directory of optimizer configs.",
)
@click.option(
    "-R",
    "--repetitions",
    required=True,
    type=int,
    help="Number of times to repeat each combination of (task, data, model, optimizer).",
)
@click.option(
    "-v",
    "--validation-interval",
    required=True,
    type=int,
    help="How many iterations between each validation run.",
)
@click.option(
    "-s",
    "--snapshot-interval",
    required=True,
    type=int,
    help="How many iterations between each saved snapshot.",
)
@click.option(
    "-b",
    "--keep-best-validation",
    required=True,
    type=str,
    help="Definition of what is considered the 'best' validation",
)
@click.option(
    "-n",
    "--num-workers",
    default=1,
    type=int,
    help="How many workers to spawn on to run jobs in parallel.",
)
@click.option(
    "-bf", "--bsub-flags", default=None, type=str, help="flags to pass to bsub"
)
@click.option(
    "--batch",
    default=False,
    type=bool,
    help="Whether to run the jobs as interactive or not.",
)
@click.option(
    "--daisy-worker",
    default=False,
    type=bool,
    help="Whether this call is a daisy worker or not.",
)
@click_config_file.configuration_option(section="runs")
def predict(
    name,
    predict_data,
    daisy_config,
    tasks,
    data,
    models,
    optimizers,
    repetitions,
    validation_interval,
    snapshot_interval,
    keep_best_validation,
    num_workers,
    bsub_flags,
    batch,
    daisy_worker,
):
    import dacapo.config
    from dacapo.predict import run_local as predict_run_local, run_remote as predict_run_remote

    if isinstance(bsub_flags, str):
        bsub_flags = bsub_flags.split()
    if num_workers > 1:
        assert any(["-P" in flag for flag in bsub_flags]), "billing must be provided"

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
        bsub_flags=bsub_flags,
        batch=batch,
    )

    desired_runs = [run for run in runs if name == run.hash]
    data = dacapo.config.DataConfig(predict_data)
    daisy_conf = dacapo.config.DaisyConfig(daisy_config)
    daisy_conf.worker = daisy_worker
    for run in desired_runs:
        if daisy_conf.num_workers > 1 and not daisy_conf.worker:
            predict_run_remote(run, data, daisy_conf)
            raise Exception("Running remotely!")
        else:
            raise Exception("Don't run locally yet!")
            predict_run_local(run, data, daisy_conf)


@cli.group()
def visualize():
    pass
