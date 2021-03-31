import zarr
from funlib.geometry import Coordinate
import funlib.run
import daisy

from dacapo.store import sanatize, MongoDbStore
from dacapo.predict import predict_one

import time
import logging
import subprocess

logger = logging.getLogger(__name__)


def validate_one(run, iteration):
    # This is a blocking call, will only return when validation
    # is completed.
    # It is assumed that we first need to predict on the validation data,
    # so we make a blocking call to predict first.
    # When that is complete, we create a daisy
    # validation task with validation workers that only need cpus.
    outdir = run.validation_outdir(iteration)
    container = "data.zarr"
    backbone_checkpoint, head_checkpoints = run.get_validation_checkpoints(iteration)

    logger.info("Predicting!")
    success = predict_one(
        run_id=run.id,
        prediction_id=f"validation_{iteration}",
        dataset_id=run.dataset,
        data_source="validate",
        out_container=outdir / container,
        backbone_checkpoint=backbone_checkpoint,
        head_checkpoints=head_checkpoints,
    )
    message = "success" if success else "failure"
    logger.info(f"Done predicting! Prediction was a {message}! returned {success}!")

    # start a new cpu job on cluster for this?
    run_validation_worker(run, iteration)


def run_validation_worker(run, iteration):
    # It is assumed that postprocessor tasks only need cpus, so
    # post processing tasks do not spawn new jobs in their
    # processing functions.
    store = MongoDbStore()
    task = store.get_task(run.task)
    pred_id = run.validation_id(iteration)
    for predictor, post_processor in zip(task.predictors, task.post_processors):
        out_dir = run.validation_outdir(iteration)
        validation_tasks, parameters = post_processor.tasks(
            pred_id=pred_id,
            container=out_dir / "data.zarr",
            input_dataset=f"volumes/{predictor.name}",
            output_dataset=f"volumes/{predictor.name}_post_processed_{{parameters}}",
        )
        success = daisy.run_blockwise(validation_tasks)

        if success:
            for i, parameters in enumerate(parameters):
                post_processed_dataset = f"volumes/{predictor.name}_post_processed_{i}"
                post_processed = daisy.open_ds(
                    out_dir / "data.zarr", post_processed_dataset
                )

                raise Exception("Evaluators not yet running blockwise!")

                # gt_dataset can be stored in any format (csv, ...). may be too large to load
                # into memory. How can we get a blockwise processable dataset out of this?

                # Once, at the start of a run, we build our gunpowder pipeline to
                # retrieve gt data and write to zarr. Then open as daisy array here.

                # something like:
                gt = daisy.open_ds(run.validation_container, run.gt_dataset)

                # Seperate evaluation into an Evaluator
                # Should operate blockwise like everything else.
                predictor.evaluate(pred_id, post_processed, gt, parameters, i)
        else:
            logger.error(
                f"Post processing for predictor: {predictor} and post_processor: {post_processor} "
                "failed. Cannot continue with evaluation!"
            )


def validate_local(run, iteration):
    # Must already be in daisy context
    # Predictions must already be done
    daisy.Client()
    if not run.outdir.exists():
        run.outdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Validating run {run.id} at iteration {iteration}:")
    validate(run, iteration)


def validate_remote(run, iteration):
    if not run.validation_outdir(iteration).exists():
        run.validation_outdir(iteration).mkdir(parents=True, exist_ok=True)

    command_args = funlib.run.run(
        command=f"dacapo validate-one -r {run.id} -i {iteration}",
        num_cpus=5,
        num_gpus=1,
        queue="gpu_rtx",
        execute=False,
        expand=False,
        flags=run.execution_details.bsub_flags.split(" "),
        batch=run.execution_details.batch,
        log_file=f"{run.validation_outdir(iteration)}/log.out",
        error_file=f"{run.validation_outdir(iteration)}/log.err",
    )
    subprocess.Popen(" ".join(command_args), shell=True, encoding="UTF-8")


def validate(run, iteration):
    raise Exception("Post processing/evaluation not yet implemented!")
    for ret in predictor.evaluate(
        ds["prediction"], ds["gt"], ds["target"], store_best_result
    ):

        if store_best_result:
            parameters, scores, results = ret
            score = scores["average"][best_score_name]
            if best_score is None or best_score_relation(score, best_score) == score:
                best_score = score
                best_parameters = parameters
                best_results = results
        else:
            parameters, scores = ret

        all_scores[str(parameters.id)] = {
            "post_processing_parameters": parameters.to_dict(),
            "scores": scores,
        }

    if store_best_result:
        f = zarr.open(f"{out_dir / out_filename}")
        for k, v in best_results.items():
            f[k] = v.data
            f[k].attrs["offset"] = v.roi.get_offset()
            f[k].attrs["resolution"] = v.voxel_size
        d = sanatize(best_parameters.to_dict())
        for k, v in d.items():
            f.attrs[k] = v

    return all_scores


def save_validation_results():
    logger.warning("validating")
    scores = validate(
        task=task,
        dataset=dataset,
        model=model,
        out_dir=self.outdir / "validations",
        out_filename=f"validate_{i}.zarr",
        backbone_checkpoint=backbone_checkpoint,
        head_checkpoints=head_checkpoints,
        best_score_name=self.execution_details.best_score_name,
        best_score_relation=self.execution_details.best_score_relation,
        store_best_result=True,
    )

    self.validation_scores.add_validation_iteration(i, scores)
    store.store_validation_scores(self)

    if self.execution_details.best_score_name is not None:

        # get best sample-average score for each iteration over
        # all post-processing parameters
        best_iteration_scores = np.array(
            [
                self.execution_details.best_score_relation(
                    [
                        v["scores"]["average"][self.execution_details.best_score_name]
                        for v in parameter_scores.values()
                    ]
                )
                for parameter_scores in self.validation_scores.scores
            ],
            dtype=np.float32,
        )

        # replace nan
        replace = -self.execution_details.best_score_relation([-np.inf, np.inf])
        isnan = np.isnan(best_iteration_scores)
        best_iteration_scores[isnan] = replace

        # get best score over all iterations
        best = self.execution_details.best_score_relation(best_iteration_scores)