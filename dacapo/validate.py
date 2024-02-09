from .predict import predict
from .compute_context import LocalTorch, ComputeContext
from .experiments import Run, ValidationIterationScores
from .experiments.datasplits.datasets.arrays import ZarrArray
from .store import (
    create_array_store,
    create_config_store,
    create_stats_store,
    create_weights_store,
)

import torch

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def validate(
    run_name: str, iteration: int, compute_context: ComputeContext = LocalTorch()
):
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
    run.validation_scores.scores = stats_store.retrieve_validation_iteration_scores(
        run_name
    )

    # create weights store and read weights
    weights_store = create_weights_store()
    weights_store.retrieve_weights(run, iteration)

    return validate_run(run, iteration, compute_context=compute_context)


def validate_run(
    run: Run, iteration: int, compute_context: ComputeContext = LocalTorch()
):
    """Validate an already loaded run at the given iteration. This does not
    load the weights of that iteration, it is assumed that the model is already
    loaded correctly. Returns the best parameters and scores for this
    iteration."""
    # set benchmark flag to True for performance
    torch.backends.cudnn.benchmark = True
    run.model.eval()

    if (
        run.datasplit.validate is None
        or len(run.datasplit.validate) == 0
        or run.datasplit.validate[0].gt is None
    ):
        logger.info("Cannot validate run %s. Continuing training!", run.name)
        return None, None

    # get array and weight store
    weights_store = create_weights_store()
    array_store = create_array_store()
    iteration_scores = []

    # get post processor and evaluator
    post_processor = run.task.post_processor
    evaluator = run.task.evaluator

    # Initialize the evaluator with the best scores seen so far
    evaluator.set_best(run.validation_scores)

    for validation_dataset in run.datasplit.validate:
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
        if (
            not Path(
                f"{input_raw_array_identifier.container}/{input_raw_array_identifier.dataset}"
            ).exists()
            or not Path(
                f"{input_gt_array_identifier.container}/{input_gt_array_identifier.dataset}"
            ).exists()
        ):
            logger.info("Copying validation inputs!")
            input_voxel_size = validation_dataset.raw.voxel_size
            output_voxel_size = run.model.scale(input_voxel_size)
            input_shape = run.model.eval_input_shape
            input_size = input_voxel_size * input_shape
            output_shape = run.model.compute_output_shape(input_shape)[1]
            output_size = output_voxel_size * output_shape
            context = (input_size - output_size) / 2
            output_roi = validation_dataset.gt.roi

            input_roi = (
                output_roi.grow(context, context)
                .snap_to_grid(validation_dataset.raw.voxel_size, mode="grow")
                .intersect(validation_dataset.raw.roi)
            )
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
            input_gt = ZarrArray.create_from_array_identifier(
                input_gt_array_identifier,
                validation_dataset.gt.axes,
                output_roi,
                validation_dataset.gt.num_channels,
                validation_dataset.gt.voxel_size,
                validation_dataset.gt.dtype,
                name=f"{run.name}_validation_gt",
                write_size=output_size,
            )
            input_gt[output_roi] = validation_dataset.gt[output_roi]
        else:
            logger.info("validation inputs already copied!")

        prediction_array_identifier = array_store.validation_prediction_array(
            run.name, iteration, validation_dataset
        )
        predict(
            run.model,
            validation_dataset.raw,
            prediction_array_identifier,
            compute_context=compute_context,
            output_roi=validation_dataset.gt.roi,
        )

        post_processor.set_prediction(prediction_array_identifier)

        dataset_iteration_scores = []

        # set up dict for overall best scores
        overall_best_scores = {}
        for criterion in run.validation_scores.criteria:
            overall_best_scores[criterion] = evaluator.get_overall_best(
                validation_dataset,
                criterion,
                run.validation_scores.evaluation_scores.higher_is_better(criterion),
            )

        for parameters in post_processor.enumerate_parameters():
            output_array_identifier = array_store.validation_output_array(
                run.name, iteration, parameters, validation_dataset
            )

            post_processed_array = post_processor.process(
                parameters, output_array_identifier
            )

            scores = evaluator.evaluate(output_array_identifier, validation_dataset.gt)
            for criterion in run.validation_scores.criteria:
                # replace predictions in array with the new better predictions
                if evaluator.is_best(
                    validation_dataset,
                    parameters,
                    criterion,
                    scores,
                ):
                    # then this is the current best score for this parameter, but not necessarily the overall best
                    higher_is_better = scores.higher_is_better(criterion)
                    # initial_best_score = overall_best_scores[criterion]
                    current_score = getattr(scores, criterion)
                    if not overall_best_scores[criterion] or (
                        (
                            higher_is_better
                            and current_score > overall_best_scores[criterion]
                        )
                        or (
                            not higher_is_better
                            and current_score < overall_best_scores[criterion]
                        )
                    ):
                        overall_best_scores[criterion] = current_score

                        # For example, if parameter 2 did better this round than it did in other rounds, but it was still worse than parameter 1
                        # the code would have overwritten it below since all parameters write to the same file. Now each parameter will be its own file
                        # Either we do that, or we only write out the overall best, regardless of parameters
                        best_array_identifier = array_store.best_validation_array(
                            run.name, criterion, index=validation_dataset.name
                        )
                        best_array = ZarrArray.create_from_array_identifier(
                            best_array_identifier,
                            post_processed_array.axes,
                            post_processed_array.roi,
                            post_processed_array.num_channels,
                            post_processed_array.voxel_size,
                            post_processed_array.dtype,
                        )
                        best_array[best_array.roi] = post_processed_array[
                            post_processed_array.roi
                        ]
                        best_array.add_metadata(
                            {
                                "iteration": iteration,
                                criterion: getattr(scores, criterion),
                                "parameters_id": parameters.id,
                            }
                        )
                        weights_store.store_best(
                            run, iteration, validation_dataset.name, criterion
                        )

            # delete current output. We only keep the best outputs as determined by
            # the evaluator
            # remove deletion for now since we have to rerun later to double check things anyway
            # array_store.remove(output_array_identifier)

            dataset_iteration_scores.append(
                [getattr(scores, criterion) for criterion in scores.criteria]
            )

        iteration_scores.append(dataset_iteration_scores)
        array_store.remove(prediction_array_identifier)

    run.validation_scores.add_iteration_scores(
        ValidationIterationScores(iteration, iteration_scores)
    )
    stats_store = create_stats_store()
    stats_store.store_validation_iteration_scores(run.name, run.validation_scores)
