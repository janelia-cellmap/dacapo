from dacapo.store.create_store import create_array_store
from .experiments import Run
from .compute_context import LocalTorch
from .store import create_config_store, create_stats_store, create_weights_store
from .validate import validate_run

import torch
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


def train(run_name, compute_context=LocalTorch()):
    if compute_context.train(run_name):
        # if compute context runs train in some other process
        # we are done here.
        return

    logger.info("Starting/resuming training for run %s...", run_name)

    # create run

    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)

    # read in previous training/validation stats

    stats_store = create_stats_store()
    run.training_stats = stats_store.retrieve_training_stats(run_name)
    run.validation_scores.iteration_scores = stats_store.retrieve_validation_iteration_scores(run_name)

    train_until = run_config.num_iterations
    trained_until = run.training_stats.trained_until()
    validation_interval = run_config.validation_interval

    logger.info("Current state: trained until %d/%d", trained_until, train_until)

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

            raise RuntimeError(
                f"Found weights for iteration {latest_weights_iteration}, but "
                f"run {run.name} was only trained until {trained_until}."
            )

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
    run.trainer.set_iteration(trained_until)
    run.trainer.build_batch_provider(
        run.datasplit.train,
        run.model,
        run.task,
        array_store.snapshot_container(run.name),
    )

    with run.trainer as trainer:
        while trained_until < train_until:

            # train for at most 100 iterations at a time, then store training stats
            iterations = min(100, train_until - trained_until)
            iteration_stats = None

            for iteration_stats in tqdm(
                trainer.iterate(
                    iterations,
                    run.model,
                    run.optimizer,
                    compute_context.device,
                ),
                "training",
                iterations,
            ):

                run.training_stats.add_iteration_stats(iteration_stats)

                if (iteration_stats.iteration + 1) % validation_interval == 0:
                    break

            if iteration_stats is None or not (
                (iteration_stats.iteration + 1) % validation_interval == 0
            ):
                stats_store.store_training_stats(run_name, run.training_stats)
                trained_until = run.training_stats.trained_until()
                continue

            run.model.eval()
            # free up optimizer memory to allow larger validation blocks
            run.model = run.model.to(torch.device("cpu"))
            run.move_optimizer(torch.device("cpu"), empty_cuda_cache=True)

            weights_store.store_weights(run, iteration_stats.iteration + 1)
            validate_run(
                run,
                iteration_stats.iteration + 1,
                compute_context=compute_context,
            )
            stats_store.store_validation_scores(run_name, run.validation_scores)
            stats_store.store_training_stats(run_name, run.training_stats)

            # make sure to move optimizer back to the correct device
            run.move_optimizer(compute_context.device)
            run.model.train()

            weights_store.store_weights(
                run, run.training_stats.trained_until(), remove_old=True
            )
            stats_store.store_training_stats(run_name, run.training_stats)
            trained_until = run.training_stats.trained_until()

    logger.info("Trained until %d, finished.", trained_until)
