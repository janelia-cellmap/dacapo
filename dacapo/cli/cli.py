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
    "-i",
    "--num-iterations",
    required=True,
    type=int,
    help="Number of iterations to train.",
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
    num_iterations,
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

    bsub_flags = bsub_flags.split(" ")
    if num_workers > 1:
        assert any(["-P" in flag for flag in bsub_flags]), "billing must be provided"

    runs = dacapo.enumerate_runs(
        task_configs=task_configs,
        data_configs=data_configs,
        model_configs=model_configs,
        optimizer_configs=optimizer_configs,
        repetitions=repetitions,
        num_iterations=num_iterations,
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
    "-i",
    "--num-iterations",
    required=True,
    type=int,
    help="Number of iterations to train.",
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
    num_iterations,
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
        int(num_iterations),
        int(validation_interval),
        int(snapshot_interval),
        keep_best_validation,
    )
    dacapo.run_local(run)


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
    "-i",
    "--num-iterations",
    required=True,
    type=int,
    help="Number of iterations to train.",
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
def clear_all(
    tasks,
    data,
    models,
    optimizers,
    repetitions,
    num_iterations,
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

    bsub_flags = bsub_flags.split(" ")
    if num_workers > 1:
        assert any(["-P" in flag for flag in bsub_flags]), "billing must be provided"

    runs = dacapo.enumerate_runs(
        task_configs=task_configs,
        data_configs=data_configs,
        model_configs=model_configs,
        optimizer_configs=optimizer_configs,
        repetitions=repetitions,
        num_iterations=num_iterations,
        validation_interval=validation_interval,
        snapshot_interval=snapshot_interval,
        keep_best_validation=keep_best_validation,
        bsub_flags=bsub_flags,
        batch=batch,
    )

    store = dacapo.store.MongoDbStore()
    for run in runs:
        output_dir = run.outdir
        if output_dir.exists():
            saved_checkpoints = []
            for f in output_dir.iterdir():
                parts = f.name.split(".")
                if len(parts) > 1 and parts[-1] == "checkpoint":
                    try:
                        saved_checkpoints.append(int(parts[0]))
                    except ValueError:
                        f.unlink()
                elif f.is_file():
                    f.unlink()
                elif f.is_dir():
                    import shutil

                    shutil.rmtree(f)

            for checkpoint in saved_checkpoints:
                if checkpoint == max(saved_checkpoints):
                    pass
                else:
                    (output_dir / f"{checkpoint}.checkpoint").unlink()

        deleted_runs = list(store.runs.find({"hash": run.hash}))
        assert len(deleted_runs) <= 1
        for run_data in deleted_runs:
            run_id = run_data["id"]
            deleted = store.runs.delete_one({"hash": run.hash})
            print(f"deleted run: {deleted.deleted_count} runs")
            deleted = store.training_stats.delete_many({"run": run_id})
            print(f"deleted training_stats: {deleted.deleted_count} training_stats")
            deleted = store.validation_scores.delete_many({"run": run_id})
            print(
                f"deleted validation_scores: {deleted.deleted_count} validation scores"
            )

        task_id = run.task_config.to_dict()["id"]
        deleted_tasks = list(store.tasks.find({"id": task_id}))
        assert len(deleted_tasks) <= 1
        for task_data in deleted_tasks:
            store.tasks.delete_one({"id": task_id})
            print(f"deleted task: {task_id}")


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
    help="The config file for daisy.",
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
    "-i",
    "--num-iterations",
    required=True,
    type=int,
    help="Number of iterations to train.",
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
def predict(
    name,
    predict_data,
    daisy_config,
    tasks,
    data,
    models,
    optimizers,
    repetitions,
    num_iterations,
    validation_interval,
    snapshot_interval,
    keep_best_validation,
    num_workers,
    bsub_flags,
    batch,
):
    import dacapo.config
    from dacapo.data import Data
    from dacapo.tasks import Task
    import daisy
    from dacapo.predict import (
        run_local as predict_run_local,
        run_remote as predict_run_remote,
    )

    try:
        daisy.Client()
        daisy_worker = True
    except KeyError:
        daisy_worker = False

    daisy_conf = dacapo.config.DaisyConfig(daisy_config)
    daisy_conf.worker = daisy_worker

    if isinstance(bsub_flags, str):
        bsub_flags = bsub_flags.split()
    if daisy_conf.num_workers > 1 and not daisy_conf.worker:
        # start daisy workers who call dacapo.predict with daisy_worker = True

        dacapo_flags = (
            f"--name {name} "
            f"--predict-data {predict_data} "
            f"--daisy-config {daisy_config} "
            f"--tasks {tasks} "
            f"--data {data} "
            f"--models {models} "
            f"--optimizers {optimizers} "
            f"--repetitions {repetitions} "
            f"--num-iterations {num_iterations} "
            f"--validation-interval {validation_interval} "
            f"--snapshot-interval {snapshot_interval} "
            f"--keep-best-validation {keep_best_validation} "
        )

        prediction_data = dacapo.config.DataConfig(predict_data)

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
            num_iterations=num_iterations,
            validation_interval=validation_interval,
            snapshot_interval=snapshot_interval,
            keep_best_validation=keep_best_validation,
            bsub_flags=bsub_flags,
            batch=batch,
        )

        desired_runs = [run for run in runs if name == run.hash]

        for run in desired_runs:
            predict_run_remote(
                run,
                prediction_data,
                daisy_conf,
                dacapo_flags,
                bsub_flags,
            )

    else:

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
            num_iterations=num_iterations,
            validation_interval=validation_interval,
            snapshot_interval=snapshot_interval,
            keep_best_validation=keep_best_validation,
            bsub_flags=bsub_flags,
            batch=batch,
        )

        desired_runs = [run for run in runs if name == run.hash]
        prediction_data = dacapo.config.DataConfig(predict_data)
        for run in desired_runs:
            predict_run_local(run, prediction_data, daisy_conf)


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
    "-i",
    "--num-iterations",
    required=True,
    type=int,
    help="Number of iterations to train.",
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
def visualize(
    tasks,
    data,
    models,
    optimizers,
    repetitions,
    num_iterations,
    validation_interval,
    snapshot_interval,
    keep_best_validation,
    num_workers,
    bsub_flags,
    batch,
):
    import dacapo.config

    print("Finding runs...")
    tasks = dacapo.config.find_task_configs(tasks)
    datas = dacapo.config.find_data_configs(data)
    models = dacapo.config.find_model_configs(models)
    optimizers = dacapo.config.find_optimizer_configs(optimizers)
    runs = dacapo.enumerate_runs(
        tasks,
        datas,
        models,
        optimizers,
        repetitions=repetitions,
        num_iterations=num_iterations,
        validation_interval=validation_interval,
        snapshot_interval=snapshot_interval,
        keep_best_validation=keep_best_validation,
        bsub_flags=bsub_flags,
        batch=batch,
    )

    print(
        f"Visualizing tasks: {[t.hash for t in tasks]}\n"
        f"datas: {[d.hash for d in datas]}\n"
        f"models: {[m.hash for m in models]}\n"
        f"optimizers: {[o.hash for o in optimizers]}"
    )

    print("Fetching data...")
    store = dacapo.store.MongoDbStore()
    for run in runs:
        store.sync_run(run)
        store.read_training_stats(run)
        store.read_validation_scores(run)

    print("Plotting...")
    dacapo.analyze.plot_runs(runs, smooth=100, validation_score=keep_best_validation)

    # import numpy as np

    # def get_best(self, score_name=None, higher_is_better=True):

    #     names = self.get_score_names()

    #     best_scores = {name: [] for name in names}
    #     for iteration_scores in self.scores:
    #         ips = np.array(
    #             [
    #                 parameter_scores["scores"]["average"][score_name]
    #                 for parameter_scores in iteration_scores.values()
    #             ]
    #         )
    #         print(ips[:10])
    #         print(np.isnan(ips[:10]))
    #         ips[np.isnan(ips)] = -np.inf if higher_is_better else np.inf
    #         print(ips[:10])
    #         i = np.argmax(ips) if higher_is_better else np.argmin(ips)
    #         for name in names:
    #             best_scores[name].append(
    #                 list(iteration_scores.values())[i]["scores"]["average"][name]
    #             )
    #     return best_scores

    # best = get_best(run.validation_scores, "fscore")
    # print(best["detection_fscore"])
