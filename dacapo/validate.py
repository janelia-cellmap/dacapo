from .predict import predict
from .experiments import Run, ValidationIterationScores
from .experiments.datasplits.datasets.arrays import ZarrArray
from .store.create_store import (
    create_array_store,
    create_config_store,
    create_stats_store,
    create_weights_store,
)

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def validate(
    run_name: str,
    iteration: int,
    num_workers: int = 1,
    output_dtype: str = "uint8",
    overwrite: bool = True,
):
    """
    Validate a run at a given iteration. Loads the weights from a previously
    stored checkpoint. Returns the best parameters and scores for this
    iteration.
    
    Args:
        run_name: The name of the run to validate. 
        iteration: The iteration to validate.
        num_workers: The number of workers to use for validation.
        output_dtype: The dtype to use for the output arrays.
        overwrite: Whether to overwrite existing output arrays
    Returns:
        The best parameters and scores for this iteration
    Raises:
        ValueError: If the run does not have a validation dataset or the dataset does not have ground truth.
    Example:
        validate("my_run", 1000)
    """

    print(f"Validating run {run_name} at iteration {iteration}...")

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

    return validate_run(
        run,
        iteration,
        num_workers=num_workers,
        output_dtype=output_dtype,
        overwrite=overwrite,
    )


# @reloading  # allows us to fix validation bugs without interrupting training
def validate_run(
    run: Run,
    iteration: int,
    num_workers: int = 1,
    output_dtype: str = "uint8",
    overwrite: bool = True,
):
    """
    Validate an already loaded run at the given iteration. This does not
    load the weights of that iteration, it is assumed that the model is already
    loaded correctly. Returns the best parameters and scores for this
    iteration.
    
    Args:
        run: The run to validate.
        iteration: The iteration to validate.
        num_workers: The number of workers to use for validation.
        output_dtype: The dtype to use for the output arrays.
        overwrite: Whether to overwrite existing output arrays
    Returns:
        The best parameters and scores for this iteration
    Raises:
        ValueError: If the run does not have a validation dataset or the dataset does not have ground truth.
    Example:
        validate_run(run, 1000)
    """

    if (
        run.datasplit.validate is None
        or len(run.datasplit.validate) == 0
        or run.datasplit.validate[0].gt is None
    ):
        raise ValueError(f"Cannot validate run {run.name} at iteration {iteration}.")

    # get array and weight store
    array_store = create_array_store()
    iteration_scores = []

    # get post processor and evaluator
    post_processor = run.task.post_processor
    evaluator = run.task.evaluator

    # Initialize the evaluator with the best scores seen so far
    try:
        evaluator.set_best(run.validation_scores)
    except ValueError:
        logger.warn(
            f"Could not set best scores for run {run.name} at iteration {iteration}."
        )

    for validation_dataset in run.datasplit.validate:
        if validation_dataset.gt is None:
            logger.error(
                "We do not yet support validating on datasets without ground truth"
            )
            raise NotImplementedError

        print(f"Validating run {run.name} on dataset {validation_dataset.name}")

        (
            input_raw_array_identifier,
            input_gt_array_identifier,
        ) = array_store.validation_input_arrays(run.name, validation_dataset.name)

        input_voxel_size = validation_dataset.raw.voxel_size
        output_voxel_size = run.model.scale(input_voxel_size)
        input_shape = run.model.eval_input_shape
        input_size = input_voxel_size * input_shape
        output_shape = run.model.compute_output_shape(input_shape)[1]
        output_size = output_voxel_size * output_shape
        context = (input_size - output_size) / 2
        output_roi = validation_dataset.gt.roi

        if (
            not Path(
                f"{input_raw_array_identifier.container}/{input_raw_array_identifier.dataset}"
            ).exists()
            or not Path(
                f"{input_gt_array_identifier.container}/{input_gt_array_identifier.dataset}"
            ).exists()
        ):
            print("Copying validation inputs!")

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
            input_raw[input_roi] = validation_dataset.raw[input_roi].squeeze()
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
            input_gt[output_roi] = validation_dataset.gt[output_roi].squeeze()
        else:
            print("validation inputs already copied!")

        prediction_array_identifier = array_store.validation_prediction_array(
            run.name, iteration, validation_dataset.name
        )
        predict(
            run,
            iteration,
            input_container=input_raw_array_identifier.container,
            input_dataset=input_raw_array_identifier.dataset,
            output_path=prediction_array_identifier,
            output_roi=validation_dataset.gt.roi,  # type: ignore
            num_workers=num_workers,
            output_dtype=output_dtype,
            overwrite=overwrite,
        )

        print(f"Predicted on dataset {validation_dataset.name}")

        post_processor.set_prediction(prediction_array_identifier)

        # # set up dict for overall best scores per dataset
        # overall_best_scores = {}
        # for criterion in run.validation_scores.criteria:
        #     overall_best_scores[criterion] = evaluator.get_overall_best(
        #         validation_dataset, criterion
        #     )

        # any_overall_best = False
        output_array_identifiers = []
        dataset_iteration_scores = []
        for parameters in post_processor.enumerate_parameters():
            output_array_identifier = array_store.validation_output_array(
                run.name, iteration, str(parameters), validation_dataset.name
            )
            output_array_identifiers.append(output_array_identifier)
            post_processed_array = post_processor.process(
                parameters, output_array_identifier
            )

            try:
                scores = evaluator.evaluate(
                    output_array_identifier, validation_dataset.gt  # type: ignore
                )
                dataset_iteration_scores.append(
                    [getattr(scores, criterion) for criterion in scores.criteria]
                )
                # for criterion in run.validation_scores.criteria:
                #     # replace predictions in array with the new better predictions
                #     if evaluator.is_best(
                #         validation_dataset,
                #         parameters,
                #         criterion,
                #         scores,
                #     ):
                #         # then this is the current best score for this parameter, but not necessarily the overall best
                #         # initial_best_score = overall_best_scores[criterion]
                #         current_score = getattr(scores, criterion)
                #         if not overall_best_scores[criterion] or evaluator.compare(
                #             current_score, overall_best_scores[criterion], criterion
                #         ):
                #             any_overall_best = True
                #             overall_best_scores[criterion] = current_score

                #             # For example, if parameter 2 did better this round than it did in other rounds, but it was still worse than parameter 1
                #             # the code would have overwritten it below since all parameters write to the same file. Now each parameter will be its own file
                #             # Either we do that, or we only write out the overall best, regardless of parameters
                #             best_array_identifier = array_store.best_validation_array(
                #                 run.name,
                #                 criterion,
                #                 index=validation_dataset.name,
                #             )
                #             best_array = ZarrArray.create_from_array_identifier(
                #                 best_array_identifier,
                #                 post_processed_array.axes,
                #                 post_processed_array.roi,
                #                 post_processed_array.num_channels,
                #                 post_processed_array.voxel_size,
                #                 post_processed_array.dtype,
                #                 output_size,
                #             )
                #             best_array[best_array.roi] = post_processed_array[
                #                 post_processed_array.roi
                #             ]
                #             best_array.add_metadata(
                #                 {
                #                     "iteration": iteration,
                #                     criterion: getattr(scores, criterion),
                #                     "parameters_id": parameters.id,
                #                 }
                #             )
                #             weights_store.store_best(
                #                 run.name,
                #                 iteration,
                #                 validation_dataset.name,
                #                 criterion,
                #             )
            except:
                logger.error(
                    f"Could not evaluate run {run.name} on dataset {validation_dataset.name} with parameters {parameters}.",
                    exc_info=True,
                    stack_info=True,
                )

        # if not any_overall_best:
        #     # We only keep the best outputs as determined by the evaluator
        #     for output_array_identifier in output_array_identifiers:
        #         array_store.remove(prediction_array_identifier)
        #         array_store.remove(output_array_identifier)

        iteration_scores.append(dataset_iteration_scores)

    run.validation_scores.add_iteration_scores(
        ValidationIterationScores(iteration, iteration_scores)
    )
    stats_store = create_stats_store()
    stats_store.store_validation_iteration_scores(run.name, run.validation_scores)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str)
    parser.add_argument("iteration", type=int)
    args = parser.parse_args()

    validate(args.run_name, args.iteration)
