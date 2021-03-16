import zarr
from funlib.geometry import Coordinate
import funlib.run

from .daisy import predict
from dacapo.store import sanatize, MongoDbStore

import time


def validate_one(run, iteration):
    print(f"Validating iteration {iteration}")

    if run.execution_details.num_workers is not None:

        from multiprocessing import Pool

        # TODO: do we want to support multiple workers to validate
        # on large volumes?.
        with Pool(1) as pool:
            pool.starmap(validate_remote, zip([run], [iteration]))

    else:
        validate_local(run, iteration)


def validate_local(run, iteration):
    if not run.outdir.exists():
        run.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Validating run {run.id} at iteration {iteration}:")
    validate(run, iteration)


def validate_remote(run, iteration):
    print(f"synced run with id {run.id}")
    if not run.validation_outdir(iteration).exists():
        run.validation_outdir(iteration).mkdir(parents=True, exist_ok=True)

    funlib.run.run(
        command=f"dacapo validate-one -r {run.id} -i {iteration}",
        num_cpus=5,
        num_gpus=1,
        queue="gpu_rtx",
        execute=True,
        flags=run.execution_details.bsub_flags.split(" "),
        batch=run.execution_details.batch,
        log_file=f"{run.validation_outdir(iteration)}/log.out",
        error_file=f"{run.validation_outdir(iteration)}/log.err",
    )


def validate(run, iteration):
    store = MongoDbStore()
    task = store.get_task(run.task)
    dataset = store.get_dataset(run.dataset)
    model = store.get_model(run.model)

    input_shape = Coordinate(model.input_shape)
    output_shape = (
        Coordinate(model.output_shape) if model.output_shape is not None else None
    )

    backbone_checkpoint, head_checkpoints = run.get_validation_checkpoints(iteration)

    print("Predicting on validation data...")
    start = time.time()
    ds = predict(
        job_id=f"{run.id}_{iteration}",
        task=task,
        dataset=dataset,
        model=model,
        input_shape=input_shape,
        output_shape=output_shape,
        output_dir=run.validation_outdir(iteration),
        output_filename="data.zarr",
        backbone_checkpoint=backbone_checkpoint,
        head_checkpoints=head_checkpoints,
        raw=dataset.raw.validate,
        gt=dataset.gt.validate,
    )
    print(f"...done ({time.time() - start}s)")

    all_scores = {}
    best_score = None
    best_parameters = None
    best_results = None
    raise Exception("Post processing/evaluation not yet supported!")
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
    print("validating")
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