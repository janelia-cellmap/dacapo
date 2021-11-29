from .predict import predict
from .compute_context import LocalTorch
from .experiments import Run, ValidationIterationScores
from .experiments.datasplits.datasets.arrays import ZarrArray
from .store import (
    create_array_store,
    create_config_store,
    create_stats_store,
    create_weights_store,
)

import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def validate(run_name, iteration, compute_context = LocalTorch()):
    """Validate a run at a given iteration. Loads the weights from a previously
    stored checkpoint. Returns the best parameters and scores for this
    iteration."""

    logger.info("Validating run %s at iteration %d...", run_name, iteration)

    # create run

    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)

    # read in previous training/validation stats

    stats_store = create_stats_store()
    run.training_stats = stats_store.retrieve_training_stats(run_name)
    run.validation_scores = stats_store.retrieve_validation_scores(run_name)

    # read weights for the given iteration

    weights_store = create_weights_store()
    weights_store.retrieve_weights(run, iteration)

    return validate_run(run, iteration, compute_context=compute_context)


def validate_run(run, iteration, compute_context=LocalTorch()):
    """Validate an already loaded run at the given iteration. This does not
    load the weights of that iteration, it is assumed that the model is already
    loaded correctly. Returns the best parameters and scores for this
    iteration."""

    logger.info("Validating run %s...", run.name)

    # create an array store

    array_store = create_array_store()

    # predict on validation dataset
    run.model = run.model.to(compute_context.device)

    (
        input_raw_array_identifier,
        input_gt_array_identifier,
    ) = array_store.validation_input_arrays(run.name)
    if (
        not Path(
            f"{input_raw_array_identifier.container}/{input_raw_array_identifier.dataset}"
        ).exists()
        or not Path(
            f"{input_gt_array_identifier.container}/{input_gt_array_identifier.dataset}"
        ).exists()
    ):
        logger.info("Copying validation inputs!")
        input_voxel_size = run.datasplit.validate[0].raw.voxel_size
        output_voxel_size = run.model.scale(input_voxel_size)
        input_size = run.model.input_shape * input_voxel_size
        output_size = run.model.output_shape * output_voxel_size
        context = (input_size - output_size) / 2
        output_roi = run.datasplit.validate[0].gt.roi
        input_roi = output_roi.grow(context, context).intersect(
            run.datasplit.validate[0].raw.roi
        )
        input_raw = ZarrArray.create_from_array_identifier(
            input_raw_array_identifier,
            run.datasplit.validate[0].raw.axes,
            input_roi,
            run.datasplit.validate[0].raw.num_channels,
            run.datasplit.validate[0].raw.voxel_size,
            run.datasplit.validate[0].raw.dtype,
            name=f"{run.name}_validation_raw",
        )
        input_raw[input_roi] = run.datasplit.validate[0].raw[input_roi]
        input_gt = ZarrArray.create_from_array_identifier(
            input_gt_array_identifier,
            run.datasplit.validate[0].gt.axes,
            output_roi,
            run.datasplit.validate[0].gt.num_channels,
            run.datasplit.validate[0].gt.voxel_size,
            run.datasplit.validate[0].gt.dtype,
            name=f"{run.name}_validation_gt",
        )
        input_gt[output_roi] = run.datasplit.validate[0].gt[output_roi]
    else:
        logger.info("validation inputs already copied!")

    prediction_array_identifier = array_store.validation_prediction_array(
        run.name,
        iteration)
    predict(
        run.model,
        run.datasplit.validate[0].raw,
        prediction_array_identifier,
        compute_context=compute_context)

    # post-process and evaluate for each parameter

    post_processor = run.task.post_processor
    evaluator = run.task.evaluator
    iteration_scores = ValidationIterationScores(iteration, [])

    post_processor.set_prediction(prediction_array_identifier)

    # there can be multiple best iteration/parameter predictions
    # The evaluator stores a record of them internally here
    evaluator.set_best(run.validation_scores)

    for parameters in post_processor.enumerate_parameters():

        output_array_identifier = array_store.validation_output_array(
            run.name, iteration, parameters
        )

        post_processed_array = post_processor.process(
            parameters, output_array_identifier
        )

        scores = evaluator.evaluate(
            output_array_identifier, run.datasplit.validate[0].gt
        )

        replaced = evaluator.is_best(iteration, parameters, scores, None)
        for criterion in replaced:
            # replace predictions in array with the new better predictions
            best_array_identifier = array_store.best_validation_array(
                run.name, criterion
            )
            best_array = ZarrArray.create_from_array_identifier(
                best_array_identifier,
                post_processed_array.axes,
                post_processed_array.roi,
                post_processed_array.num_channels,
                post_processed_array.voxel_size,
                post_processed_array.dtype,
            )
            best_array[best_array.roi] = post_processed_array[post_processed_array.roi]
            weights_store.store_best(run, iteration, criterion)

        # delete current output. We only keep the best outputs as determined by
        # the evaluator
            array_store.remove(output_array_identifier)
        weights_store.remove(run, iteration)

        iteration_scores.parameter_scores.append((parameters, scores))

    array_store.remove(prediction_array_identifier)

    run.validation_scores.add_iteration_scores(iteration_scores)
    stats_store = create_stats_store()
    stats_store.store_validation_scores(run.name, run.validation_scores)
