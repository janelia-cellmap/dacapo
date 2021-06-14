import logging

logger = logging.getLogger(__name__)

def validate(run_id, iteration):
    logger.info("Validating run %s at iteration %d...", run_id, iteration)
