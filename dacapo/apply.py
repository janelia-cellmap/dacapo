import logging

logger = logging.getLogger(__name__)


def apply(run_name: str, iteration: int, dataset_name: str):
    logger.info(
        "Applying results from run %s at iteration %d to dataset %s",
        run_name,
        iteration,
        dataset_name,
    )
