from dacapo.compute_context import create_compute_context
from dacapo.store.create_store import (
    create_array_store,
    create_config_store,
    create_stats_store,
    create_weights_store,
)
from dacapo.experiments import RunConfig
from dacapo.validate import validate_run
from dacapo.experiments.training_iteration_stats import TrainingIterationStats

import torch
from tqdm import tqdm

import logging
import time

logger = logging.getLogger(__name__)


def train(run_name: str, validate=True):
    """
    Train a run

    Args:
        run_name: Name of the run to train
    Raises:
        ValueError: If run_name is not found in config store
    Examples:
        >>> train("run_name")
    """

    # check config store to see if run is already being trained TODO
    # if ...:
    #     logger.error(f"Run {run_name} is already being trained")
    #     # if compute context runs train in some other process
    #     # we are done here.
    #     return

    print(f"Training run {run_name}")

    # create run

    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)

    return train_run(run_config, validate)


def train_run(run: RunConfig, validate: bool = True, save_snapshots: bool = False):
    """
    Train a run

    Args:
        run: Run object to train
    Raises:
        ValueError: If run_name is not found in config store

    """
    print(f"Starting/resuming training for run {run.name}...")

    stats_store = create_stats_store()
    weights_store = create_weights_store()
    array_store = create_array_store()

    start_iteration = run.resume_training(stats_store, weights_store)

    # start/resume training
    # set flag to improve training speeds
    torch.backends.cudnn.benchmark = True

    # make sure model and optimizer are on correct device.
    # loading weights directly from a checkpoint into cuda
    # can allocate twice the memory of loading to cpu before
    # moving to cuda.
    compute_context = create_compute_context()
    run.to(compute_context.device)
    logger.info(f"Training on {compute_context.device}")

    dataloader = run.data_loader()
    snapshot_container = array_store.snapshot_container(run.name)

    for i, batch in (
        bar := tqdm(
            enumerate(dataloader, start=start_iteration),
            total=run.num_iterations,
            initial=start_iteration,
            desc="training",
            postfix={"loss": None},
        )
    ):
        t_train_step = time.time()
        loss, batch_out = run.train_step(batch["raw"], batch["target"], batch["weight"])

        iteration_stats = TrainingIterationStats(
            loss=loss,
            iteration=i,
            time=time.time() - t_train_step,
        )
        run.training_stats.add_iteration_stats(iteration_stats)

        bar.set_postfix({"loss": loss})

        if (
            run.snapshot_interval is not None
            and i % run.snapshot_interval == 0
            and save_snapshots
        ):
            # save snapshot. We save the snapshot at the start of every
            # {snapshot_interval} iterations. This is for debugging
            # purposes so you get snapshots quickly.
            run.save_snapshot(i, batch, batch_out, snapshot_container)

        if i % run.validation_interval == run.validation_interval - 1 or i == run.num_iterations - 1:
            # run "end of epoch steps" such as stepping the learning rate
            # scheduler, storing stats, and writing out weights.
            try:
                run.lr_scheduler.step((i + 1) // run.validation_interval)
            except UserWarning as w:
                # TODO: What is going on here? Definitely calling optimizer.step()
                # before calling lr_scheduler.step(), but still getting a warning.
                logger.warning(w)
                pass

            # Store checkpoint and training stats
            stats_store.store_training_stats(run.name, run.training_stats)
            weights_store.store_weights(run, i + 1)

            if validate:
                # VALIDATE
                validate_run(
                    run,
                    i + 1,
                )
                stats_store.store_validation_iteration_scores(
                    run.name, run.validation_scores
                )

        if i == run.num_iterations - 1:
            break
