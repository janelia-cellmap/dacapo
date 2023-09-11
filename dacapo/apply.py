import logging
from funlib.geometry import Roi

from dacapo.experiments.tasks.post_processors.post_processor_parameters import (
    PostProcessorParameters,
)
from dacapo.store.array_store import LocalArrayIdentifier
from dacapo.predict import predict
from dacapo.compute_context import LocalTorch, ComputeContext
from dacapo.experiments import Run, ValidationIterationScores
from dacapo.experiments.datasplits.datasets.arrays import ZarrArray
from dacapo.store import (
    create_array_store,
    create_config_store,
    create_stats_store,
    create_weights_store,
)

import torch

from pathlib import Path

logger = logging.getLogger(__name__)


def apply(
    run_name: str,
    dataset_name: str,
    output_path: str,
    validation_name: str,
    roi: Roi or None = None,
    criterion: str or None = "voi",
    iteration: int or None = None,
    compute_context: ComputeContext = LocalTorch(),
):
    """Load weights and apply a model to a dataset. If iteration is None, the best iteration based on the criterion is used. If roi is None, the whole input dataset is used."""

    # create run

    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)

    # read in previous training/validation stats TODO: is this necessary?

    stats_store = create_stats_store()
    run.training_stats = stats_store.retrieve_training_stats(run_name)
    run.validation_scores.scores = stats_store.retrieve_validation_iteration_scores(
        run_name
    )

    # create weights store
    weights_store = create_weights_store()

    # load weights
    if iteration is None:
        # weights_store._load_best(run, criterion)
        iteration = weights_store.retrieve_best(run_name, validation_name, criterion)
    weights_store.retrieve_weights(run, iteration)  # shouldn't this be load_weights?

    # find the best parameters
    scores = [s for s in run.validation_scores.scores if s.iteration == iteration + 1][
        0
    ].scores
    # paremeters = ... scores[criterion]???

    # # make array identifiers for input, predictions and outputs
    # array_store = create_array_store()
    # input_array_identifier = ...
    # prediction_array_identifier = LocalArrayIdentifier(
    #     output_path, dataset_name, "prediction"...
    # )
    # output_array_identifier = LocalArrayIdentifier(
    #     output_path, dataset_name, "output", parameters...
    # )

    logger.info(
        "Applying best results from run %s at iteration %i to dataset %s",
        run.name,
        iteration,
        dataset_name,
    )
    return apply_run(
        run,
        dataset_name,
        prediction_array_identifier,
        output_array_identifier,
        parameters,
        roi,
        compute_context,
    )


def apply_run(
    run: Run,
    parameters: PostProcessorParameters,
    input_array_identifier: LocalArrayIdentifier,
    prediction_array_identifier: LocalArrayIdentifier,
    output_array_identifier: LocalArrayIdentifier,
    roi: Roi or None = None,
    compute_context: ComputeContext = LocalTorch(),
):
    """Apply the model to a dataset. If roi is None, the whole input dataset is used. Assumes model is already loaded."""

    # Find the best parameters

    # set benchmark flag to True for performance
    torch.backends.cudnn.benchmark = True
    run.model.eval()

    ...
