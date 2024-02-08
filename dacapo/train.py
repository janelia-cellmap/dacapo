from copy import deepcopy
from dacapo.store.create_store import create_array_store
from .experiments import Run
from .compute_context import LocalTorch, ComputeContext
from .store import create_config_store, create_stats_store, create_weights_store
from .validate import validate_run

import torch
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def train(run_name: str, compute_context: ComputeContext = LocalTorch()):
    """Train a run"""

    if compute_context.train(run_name):
        logger.error("Run %s is already being trained", run_name)
        # if compute context runs train in some other process
        # we are done here.
        return

    logger.info("Training run %s", run_name)

    # create run

    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)

    return train_run(run)


def train_run(
    run: Run,
    compute_context: ComputeContext = LocalTorch(),
):
    logger.info("Starting/resuming training for run %s...", run)

    # create run

    stats_store = create_stats_store()
    run.training_stats = stats_store.retrieve_training_stats(run.name)
    run.validation_scores.scores = stats_store.retrieve_validation_iteration_scores(
        run.name
    )

    trained_until = run.training_stats.trained_until()
    validated_until = run.validation_scores.validated_until()
    if validated_until > trained_until:
        logger.info(
            f"Trained until {trained_until}, but validated until {validated_until}! "
            "Deleting extra validation stats"
        )
        run.validation_scores.delete_after(trained_until)

    logger.info("Current state: trained until %d/%d", trained_until, run.train_until)

    # read weights of the latest iteration

    weights_store = create_weights_store()
    latest_weights_iteration = weights_store.latest_iteration(run)

    if trained_until > 0:
        if latest_weights_iteration is None:
            logger.warning(
                "Run %s was previously trained until %d, but no weights are "
                "stored. Will restart training from scratch.",
                run.name,
                trained_until,
            )

            trained_until = 0
            run.training_stats.delete_after(0)
            run.validation_scores.delete_after(0)

        elif latest_weights_iteration < trained_until:
            logger.warning(
                "Run %s was previously trained until %d, but the latest "
                "weights are stored for iteration %d. Will resume training "
                "from %d.",
                run.name,
                trained_until,
                latest_weights_iteration,
                latest_weights_iteration,
            )

            trained_until = latest_weights_iteration
            run.training_stats.delete_after(trained_until)
            run.validation_scores.delete_after(trained_until)
            weights_store.retrieve_weights(run, iteration=trained_until)

        elif latest_weights_iteration == trained_until:
            logger.info("Resuming training from iteration %d", trained_until)

            weights_store.retrieve_weights(run, iteration=trained_until)

        elif latest_weights_iteration > trained_until:
            weights_store.retrieve_weights(run, iteration=latest_weights_iteration)
            logger.error(
                f"Found weights for iteration {latest_weights_iteration}, but "
                f"run {run.name} was only trained until {trained_until}. "
                "Filling stats with last observed values."
            )
            last_iteration_stats = run.training_stats.iteration_stats[-1]
            for i in range(
                last_iteration_stats.iteration, latest_weights_iteration - 1
            ):
                new_iteration_stats = deepcopy(last_iteration_stats)
                new_iteration_stats.iteration = i + 1
                run.training_stats.add_iteration_stats(new_iteration_stats)
                trained_until = run.training_stats.trained_until()

    # start/resume training

    # set flag to improve training speeds
    torch.backends.cudnn.benchmark = True

    # make sure model and optimizer are on correct device.
    # loading weights directly from a checkpoint into cuda
    # can allocate twice the memory of loading to cpu before
    # moving to cuda.
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
            no_its = iteration_stats is None  # No training steps run
            validation_it = (
                iteration_stats.iteration + 1
            ) % run.validation_interval == 0
            final_it = trained_until >= run.train_until
            if no_its or (not validation_it and not final_it):
                stats_store.store_training_stats(run.name, run.training_stats)
                continue

            run.model.eval()
            # free up optimizer memory to allow larger validation blocks
            run.model = run.model.to(torch.device("cpu"))
            run.move_optimizer(torch.device("cpu"), empty_cuda_cache=True)

            stats_store.store_training_stats(run.name, run.training_stats)
            weights_store.store_weights(run, iteration_stats.iteration + 1)
            try:
                validate_run(
                    run,
                    iteration_stats.iteration + 1,
                    compute_context=compute_context,
                )
                stats_store.store_validation_iteration_scores(
                    run.name, run.validation_scores
                )
            except Exception as e:
                logger.error(
                    f"Validation failed for run {run.name} at iteration "
                    f"{iteration_stats.iteration + 1}.",
                    exc_info=e,
                )

            # make sure to move optimizer back to the correct device
            run.move_optimizer(compute_context.device)
            run.model.train()

    logger.info("Trained until %d, finished.", trained_until)
