from .predict_local import predict

from .experiments import Run, ValidationIterationScores
from dacapo.tmp import create_from_identifier, num_channels_from_array

from .store.create_store import (
    create_array_store,
    create_config_store,
    create_stats_store,
    create_weights_store,
)
import torch

from upath import UPath as Path
import logging
from dacapo.compute_context import create_compute_context

logger = logging.getLogger(__name__)


def validate(run_name: str, iteration: int = 0, datasets_config=None):
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

    return validate_run(run, iteration, datasets_config)


def validate_run(run: Run, iteration: int, datasets_config=None):
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
        logger.error("Cannot validate run %s. Continuing training!", run.name)
        return None, None

    # get array and weight store
    array_store = create_array_store()
    iteration_scores = []

    # get post processor and evaluator
    post_processor = run.task.post_processor
    evaluator = run.task.evaluator

    input_voxel_size = run.datasplit.train[0].raw.voxel_size
    output_voxel_size = run.model.scale(input_voxel_size)

    # Initialize the evaluator with the best scores seen so far
    # evaluator.set_best(run.validation_scores)
    if datasets_config is None:
        datasets = run.datasplit.validate
    else:
        if not hasattr(run.task.evaluator, "channels"):
            raise ValueError(
                f"Evaluator must have a channels attribute to validate with custom datasets, evaluator: {run.task.evaluator} is not supported yet."
            )
        from dacapo.experiments.datasplits import DataSplitGenerator

        datasplit_config = (
            DataSplitGenerator(
                "",
                datasets_config,
                input_voxel_size,
                output_voxel_size,
                targets=run.task.evaluator.channels,
            )
            .compute()
            .validate_configs
        )
        datasets = [
            validate_config.dataset_type(validate_config)
            for validate_config in datasplit_config
        ]

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
        if (
            not Path(
                f"{input_raw_array_identifier.container}/{input_raw_array_identifier.dataset}"
            ).exists()
            or not Path(
                f"{input_gt_array_identifier.container}/{input_gt_array_identifier.dataset}"
            ).exists()
        ):
            logger.info("Copying validation inputs!")

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
            input_raw = create_from_identifier(
                input_raw_array_identifier,
                validation_dataset.raw.axis_names,
                input_roi,
                num_channels_from_array(validation_dataset.raw),
                validation_dataset.raw.voxel_size,
                validation_dataset.raw.dtype,
                name=f"{run.name}_validation_raw",
                write_size=input_size,
                overwrite=True,
            )
            input_raw[input_roi] = validation_dataset.raw[input_roi].squeeze()
            input_gt = create_from_identifier(
                input_gt_array_identifier,
                validation_dataset.gt.axis_names,
                output_roi,
                num_channels_from_array(validation_dataset.gt),
                validation_dataset.gt.voxel_size,
                validation_dataset.gt.dtype,
                name=f"{run.name}_validation_gt",
                write_size=output_size,
                overwrite=True,
            )
            input_gt[output_roi] = validation_dataset.gt[output_roi]
        else:
            logger.info("validation inputs already copied!")

        prediction_array_identifier = array_store.validation_prediction_array(
            run.name, iteration, validation_dataset
        )
        predict(
            run.model,
            input_raw_array_identifier,
            prediction_array_identifier,
            output_roi=validation_dataset.gt.roi,
        )

        post_processor.set_prediction(prediction_array_identifier)

        dataset_iteration_scores = []

        for parameters in post_processor.enumerate_parameters():
            output_array_identifier = array_store.validation_output_array(
                run.name, iteration, parameters, validation_dataset
            )

            post_processed_array = post_processor.process(
                parameters, output_array_identifier
            )

            try:
                scores = evaluator.evaluate(
                    output_array_identifier,
                    validation_dataset.gt,  # type: ignore
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
                #             best_array = create_from_identifier(
                #                 best_array_identifier,
                #                 post_processed_array.axis_names,
                #                 post_processed_array.roi,
                #                 num_channels_from_array(post_processed_array),
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

            # delete current output. We only keep the best outputs as determined by
            # the evaluator
            # array_store.remove(output_array_identifier)

            dataset_iteration_scores.append(
                [getattr(scores, criterion) for criterion in scores.criteria]
            )

        iteration_scores.append(dataset_iteration_scores)
        # array_store.remove(prediction_array_identifier)

    run.validation_scores.add_iteration_scores(
        ValidationIterationScores(iteration, iteration_scores)
    )
    stats_store = create_stats_store()
    stats_store.store_validation_iteration_scores(run.name, run.validation_scores)
