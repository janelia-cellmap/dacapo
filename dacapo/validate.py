import logging

logger = logging.getLogger(__name__)


def validate(run_name, iteration):
    logger.info("Validating run %s at iteration %d...", run_name, iteration)
