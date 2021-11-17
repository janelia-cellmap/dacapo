from .predict import predict
from .compute_context import LocalTorch
from .experiments import Run, ValidationIterationScores
from .store import \
    create_array_store, \
    create_config_store, \
    create_stats_store, \
    create_weights_store
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
    best_parameters = None
    best_scores = None

    for parameters in post_processor.enumerate_parameters():

        output_array_identifier = array_store.validation_output_array(
                run.name,
                iteration,
                parameters)

        post_processor.process(
            parameters,
            output_array_identifier)

        scores = evaluator.evaluate(
            output_array_identifier,
            run.datasplit.validate[0].gt)

        if iteration_scores.is_better(
                scores,
                run.validation_score,
                run.validation_score_minimize):

            if best_parameters is not None:

                # delete previous best output
                prev_best_array = array_store.validation_output_array(
                        run.name,
                        iteration,
                        best_parameters)
                array_store.remove(prev_best_array)

            best_parameters = parameters
            best_scores = scores

        else:

            # delete current output
            array_store.remove(output_array_identifier)

        iteration_scores.parameter_scores.append((parameters, scores))

    run.validation_scores.add_iteration_scores(iteration_scores)

    return best_parameters, best_scores
