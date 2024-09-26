from .predict_crop import predict

from .experiments import Run, ValidationIterationScores
from .experiments.datasplits.datasets.arrays import ZarrArray
from dacapo.store.create_store import (
    create_array_store,
    create_config_store,
    create_stats_store,
    create_weights_store,
)
import torch

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def full_predict(run_name: str, iteration: int = 0, datasets_config=0):
    """Validate a run at a given iteration. Loads the weights from a previously
    stored checkpoint. Returns the best parameters and scores for this
    iteration."""
    array_store = create_array_store()

    logger.info("Validating run %s at iteration %d...", run_name, iteration)

    # create run

    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)

    # read in previous training/validation stats

    stats_store = create_stats_store()
    run.training_stats = stats_store.retrieve_training_stats(run_name)
    run.validation_scores.scores = stats_store.retrieve_validation_iteration_scores(
        run_name
    )

    # create weights store and read weights
    if iteration > 0:
        weights_store = create_weights_store()
        weights_store.retrieve_weights(run, iteration)


    # set benchmark flag to True for performance
    torch.backends.cudnn.benchmark = True
    run.model.eval()

    if (run.datasplit.validate is None
        or len(run.datasplit.validate) == 0
        or run.datasplit.validate[0].gt is None
    ):
        logger.error("Cannot validate run %s. Continuing training!", run.name)
        return None, None

    # get array and weight store

    validation_dataset = run.datasplit.validate[datasets_config]

    assert (
        validation_dataset.gt is not None
    ), "We do not yet support validating on datasets without ground truth"
    logger.info(
        "Validating run %s on dataset %s", run.name, validation_dataset.name
    )

        

    prediction_array_identifier = array_store.validation_prediction_array(
        run.name, iteration+21, validation_dataset
    )
    predict(
        run.model,
        validation_dataset.raw,
        prediction_array_identifier,
        output_roi=validation_dataset.raw.roi,
    )
