import logging

logger = logging.getLogger(__name__)


def predict(run_id, iteration, dataset_id):
    logger.info(
        "Predicting run %s at iteration %d on dataset %s",
        run_id,
        iteration,
        dataset_id)
