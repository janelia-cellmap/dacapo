from dacapo.compute_context import create_compute_context
from dacapo.store.create_store import (
    create_array_store,
    create_config_store,
    create_stats_store,
    create_weights_store,
)
from dacapo.experiments import Run
from dacapo.validate import validate

import torch
from tqdm import tqdm
import threading

import logging

logger = logging.getLogger(__name__)


def train(run_name: str):
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
    run = Run(run_config)

    return train_run(run)


def train_run(run: Run):
    """
    Train a run

    Args:
        run: Run object to train
    Raises:
        ValueError: If run_name is not found in config store

    """
    print(f"Starting/resuming training for run {run}...")

    # create run

    stats_store = create_stats_store()
    run.training_stats = stats_store.retrieve_training_stats(run.name)
    run.validation_scores.scores = stats_store.retrieve_validation_iteration_scores(
        run.name
    )

    trained_until = run.training_stats.trained_until()
    validated_until = run.validation_scores.validated_until()
    if validated_until > trained_until:
        print(
            f"Trained until {trained_until}, but validated until {validated_until}! "
            "Deleting extra validation stats"
        )
        run.validation_scores.delete_after(trained_until)

    print(f"Current state: trained until {trained_until}/{run.train_until}")

    # read weights of the latest iteration

    weights_store = create_weights_store()
    latest_weights_iteration = weights_store.latest_iteration(run)

    if trained_until > 0:
        if latest_weights_iteration is None:
            logger.warning(
                f"Run {run.name} was previously trained until {trained_until}, but no weights are "
                "stored. Will restart training from scratch."
            )

            trained_until = 0
            run.training_stats.delete_after(0)
            run.validation_scores.delete_after(0)

        elif latest_weights_iteration < trained_until:
            logger.warning(
                f"Run {run.name} was previously trained until {trained_until}, but the latest "
                f"weights are stored for iteration {latest_weights_iteration}. Will resume training "
                f"from {latest_weights_iteration}."
            )

            trained_until = latest_weights_iteration
            run.training_stats.delete_after(trained_until)
            run.validation_scores.delete_after(trained_until)
            weights_store.retrieve_weights(run, iteration=trained_until)

        elif latest_weights_iteration == trained_until:
            print(f"Resuming training from iteration {trained_until}")

            weights_store.retrieve_weights(run, iteration=trained_until)

        elif latest_weights_iteration > trained_until:
            weights_store.retrieve_weights(run, iteration=latest_weights_iteration)
            logger.error(
                f"Found weights for iteration {latest_weights_iteration}, but "
                f"run {run.name} was only trained until {trained_until}. "
            )

    # start/resume training

    # set flag to improve training speeds
    torch.backends.cudnn.benchmark = True

    # make sure model and optimizer are on correct device.
    # loading weights directly from a checkpoint into cuda
    # can allocate twice the memory of loading to cpu before
    # moving to cuda.
    compute_context = create_compute_context()
    run.model = run.model.to(compute_context.device)
    run.move_optimizer(compute_context.device)

    array_store = create_array_store()
    run.trainer.iteration = trained_until
    run.trainer.build_batch_provider(
        run.datasplit.train,
        run.model,
        run.task,
        array_store.snapshot_container(run.name),
    )

    with run.trainer as trainer:
        while trained_until < run.train_until:
            # train for at most 100 iterations at a time, then store training stats
            iterations = min(100, run.train_until - trained_until)
            iteration_stats = None
            bar = tqdm(
                trainer.iterate(
                    iterations,
                    run.model,
                    run.optimizer,
                    compute_context.device,
                ),
                desc=f"training until {iterations + trained_until}",
                total=run.train_until,
                initial=trained_until,
            )
            for iteration_stats in bar:
                run.training_stats.add_iteration_stats(iteration_stats)
                bar.set_postfix({"loss": iteration_stats.loss})

                if (iteration_stats.iteration + 1) % run.validation_interval == 0:
                    break

            trained_until = run.training_stats.trained_until()

            # If this is not a validation iteration or final iteration, skip validation
            # also skip for test cases where total iterations is less than validation interval
            no_its = iteration_stats is None  # No training steps run
            validation_it = (
                iteration_stats.iteration + 1
            ) % run.validation_interval == 0
            final_it = trained_until >= run.train_until
            if final_it and (trained_until < run.validation_interval):
                # Special case for tests - skip validation, but store weights
                stats_store.store_training_stats(run.name, run.training_stats)
                weights_store.store_weights(run, iteration_stats.iteration + 1)
                continue

            if no_its or (not validation_it and not final_it):
                stats_store.store_training_stats(run.name, run.training_stats)
                continue

            stats_store.store_training_stats(run.name, run.training_stats)
            weights_store.store_weights(run, iteration_stats.iteration + 1)
            try:
                # launch validation in a separate thread to avoid blocking training
                validate_thread = threading.Thread(
                    target=validate,
                    args=(run, iteration_stats.iteration + 1),
                    name=f"validate_{run.name}_{iteration_stats.iteration + 1}",
                    daemon=True,
                )
                validate_thread.start()
                # validate(
                #     run,
                #     iteration_stats.iteration + 1,
                # )

                stats_store.store_validation_iteration_scores(
                    run.name, run.validation_scores
                )
            except Exception as e:
                logger.error(
                    f"Validation failed for run {run.name} at iteration "
                    f"{iteration_stats.iteration + 1}.",
                    exc_info=e,
                )

    print(f"Trained until {trained_until}. Finished.")
