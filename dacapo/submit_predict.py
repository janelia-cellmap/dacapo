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
from dacapo.compute_context import create_compute_context
logger = logging.getLogger(__name__)


def full_predict(run_name: str, iteration: int , roi):
    """Validate a run at a given iteration. Loads the weights from a previously
    stored checkpoint. Returns the best parameters and scores for this
    iteration."""

    logger.info("Validating run %s at iteration %d...", run_name, iteration)

    # create run

    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)

    compute_context = create_compute_context()
    device = compute_context.device
    run.model.to(device)

    # read in previous training/validation stats

    stats_store = create_stats_store()
    run.training_stats = stats_store.retrieve_training_stats(run_name)
    run.validation_scores.scores = stats_store.retrieve_validation_iteration_scores(
        run_name
    )

    # create weights store and read weights
    if iteration > 0:
        weights_store = create_weights_store()
        weights = weights_store.retrieve_weights(run, iteration)
        run.model.load_state_dict(weights.model)

    return full_predict_run(run, iteration,roi)


def full_predict_run(run: Run, iteration: int,roi):
    """Validate an already loaded run at the given iteration. This does not
    load the weights of that iteration, it is assumed that the model is already
    loaded correctly. Returns the best parameters and scores for this
    iteration."""
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
    array_store = create_array_store()

    input_voxel_size = run.datasplit.train[0].raw.voxel_size
    output_voxel_size = run.model.scale(input_voxel_size)

    # Initialize the evaluator with the best scores seen so far
    # evaluator.set_best(run.validation_scores)

    datasets = run.datasplit.validate
    for validation_dataset in datasets:
        assert (
            validation_dataset.gt is not None
        ), "We do not yet support validating on datasets without ground truth"
        logger.info(
            "Validating run %s on dataset %s", run.name, validation_dataset.name
        )

        (
            input_raw_array_identifier,
            input_gt_array_identifier,
        ) = array_store.validation_input_arrays(run.name, validation_dataset.name)

        logger.info("Copying validation inputs!")
        
        input_shape = run.model.eval_input_shape
        input_size = input_voxel_size * input_shape
        output_shape = run.model.compute_output_shape(input_shape)[1]
        output_size = output_voxel_size * output_shape
        context = (input_size - output_size) / 2
        output_roi = roi

        input_roi = (
            output_roi.grow(context, context)
            .snap_to_grid(validation_dataset.raw.voxel_size, mode="grow")
            .intersect(validation_dataset.raw.roi)
        )
        input_raw_array_identifier.dataset = "tmp_"+input_raw_array_identifier.dataset 
        input_raw = ZarrArray.create_from_array_identifier(
            input_raw_array_identifier,
            validation_dataset.raw.axes,
            input_roi,
            validation_dataset.raw.num_channels,
            validation_dataset.raw.voxel_size,
            validation_dataset.raw.dtype,
            name=f"{run.name}_validation_raw",
            write_size=input_size,
        )
        input_raw[input_roi] = validation_dataset.raw[input_roi]
       
        prediction_array_identifier = array_store.validation_prediction_array(
            run.name, iteration+224, validation_dataset
        )
        predict(
            run.model,
            input_raw_array_identifier,
            prediction_array_identifier,
            output_roi=roi,
        )