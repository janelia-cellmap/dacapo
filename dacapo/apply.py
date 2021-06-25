import logging

logger = logging.getLogger(__name__)


def apply(run_name, iteration, dataset_name):
    logger.info(
        "Applying results from run %s at iteration %d to dataset %s",
        run_name,
        iteration,
        dataset_name)


def predict(model, dataset, prediction_array):
    pass
