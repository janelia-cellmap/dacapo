from .experiments import Run
from .store import create_config_store, create_stats_store
from .validate import validate_run
import logging

logger = logging.getLogger(__name__)


def train(run_name):

    logger.info("Starting/resuming training for run %s...", run_name)

    # create run

    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)

    # read in previous training/validation stats

    stats_store = create_stats_store()
    stats_store.retrieve_training_stats(run)
    stats_store.retrieve_validation_scores(run)

    train_until = run_config.num_iterations
    trained_until = run.training_stats.trained_until()
    validation_interval = run_config.validation_interval

    logger.info(
        "Current state: trained until %d/%d",
        trained_until,
        train_until)

    # start/resume training

    run.trainer.set_iteration(trained_until)

    while trained_until < train_until:

        # train for at most 100 iterations at a time, then store training stats
        iterations = min(100, train_until - trained_until)

        for iteration_stats in run.trainer.iterate(iterations):

            run.training_stats.add_iteration_stats(iteration_stats)

            if (iteration_stats.iteration + 1) % validation_interval == 0:
                validate_run(run)

        stats_store.store_training_stats(run)
        trained_until = run.training_stats.trained_until()

    logger.info("Trained until %d, finished.", trained_until)
