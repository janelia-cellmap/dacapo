import logging
from typing import Optional
from funlib.geometry import Roi
import numpy as np
from dacapo.experiments.datasplits.datasets.dataset import Dataset

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
    input_container: str,
    input_dataset: str,
    output_path: str,
    validation_dataset: Optional[str or Dataset] = None,
    criterion: Optional[str] = "voi",
    iteration: Optional[int] = None,
    roi: Optional[Roi] = None,
    num_cpu_workers: int = 4,
    output_dtype: Optional[np.dtype or torch.dtype] = np.uint8,
    compute_context: ComputeContext = LocalTorch(),
):
    """Load weights and apply a model to a dataset. If iteration is None, the best iteration based on the criterion is used. If roi is None, the whole input dataset is used."""

    assert (validation_dataset is not None and isinstance(criterion, str)) or (
        iteration is not None
    ), "Either validation_dataset and criterion, or iteration must be provided."

    # retrieving run
    logger.info("Loading run %s", run_name)
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
        iteration = weights_store.retrieve_best(run_name, validation_dataset, criterion)
    logger.info("Loading weights for iteration %i", iteration)
    weights_store.retrieve_weights(run, iteration)  # shouldn't this be load_weights?

    # find the best parameters
    if isinstance(validation_dataset, str):
        val_ds_name = validation_dataset
        validation_dataset = [
            dataset for dataset in run.datasplit.validate if dataset.name == val_ds_name
        ][0]
    logger.info("Finding best parameters for validation dataset %s", validation_dataset)
    parameters = run.task.evaluator.get_overall_best_parameters(
        validation_dataset, criterion
    )

    # make array identifiers for input, predictions and outputs
    array_store = create_array_store()
    input_array_identifier = LocalArrayIdentifier(input_container, input_dataset)
    output_container = Path(output_path, Path(input_container).name)
    prediction_array_identifier = LocalArrayIdentifier(
        output_container, f"prediction_{run_name}_{iteration}_{parameters}"
    )
    output_array_identifier = LocalArrayIdentifier(
        output_container, f"output_{run_name}_{iteration}_{parameters}"
    )

    logger.info(
        "Applying best results from run %s at iteration %i to dataset %s",
        run.name,
        iteration,
        Path(input_container, input_dataset),
    )
    return apply_run(
        run,
        parameters,
        input_array_identifier
        prediction_array_identifier,
        output_array_identifier,
        roi,
        num_cpu_workers,
        output_dtype,
        compute_context,
    )


def apply_run(
    run: Run,
    parameters: PostProcessorParameters,
    input_array_identifier: LocalArrayIdentifier,
    prediction_array_identifier: LocalArrayIdentifier,
    output_array_identifier: LocalArrayIdentifier,
    roi: Optional[Roi] = None,
    num_cpu_workers: int = 4,
    output_dtype: Optional[np.dtype or torch.dtype] = np.uint8,
    compute_context: ComputeContext = LocalTorch(),
):
    """Apply the model to a dataset. If roi is None, the whole input dataset is used. Assumes model is already loaded."""

    # render prediction dataset
    logger.info("Predicting on dataset %s", prediction_array_identifier)
    predict(run.model, input_array_identifier, prediction_array_identifier, output_roi=roi, num_cpu_workers=num_cpu_workers, output_dtype=output_dtype compute_context=compute_context)

    # post-process the output
    logger.info("Post-processing output to dataset %s", output_array_identifier)
    post_processor = run.task.post_processor
    post_processor.set_prediction(prediction_array_identifier)
    post_processed_array = post_processor.process(
                parameters, output_array_identifier
            )
    
    logger.info("Done")
    return