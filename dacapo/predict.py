import logging

logger = logging.getLogger(__name__)


def predict(run_name, iteration, dataset_name):
    logger.info(
        "Predicting run %s at iteration %d on dataset %s",
        run_name,
        iteration,
        dataset_name)
