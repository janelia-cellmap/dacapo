from .predict_pipeline import predict
from dacapo.store import sanatize
import zarr

import time
from pathlib import Path


def validate(
    data,
    model,
    predictor,
    out_dir: Path,
    out_filename: str,
    aux_tasks=None,
    best_score_name=None,
    best_score_relation=None,
    store_best_result: bool = True,
):

    raw_data = data.raw.validate
    gt_data = data.gt.validate

    print("Predicting on validation data...")
    start = time.time()
    ds = predict(
        raw_data,
        model,
        predictor,
        output_dir=out_dir,
        output_filename=out_filename,
        gt=gt_data,
        aux_tasks=aux_tasks,
        padding_mode=data.prediction_padding,
    )
    print(f"...done ({time.time() - start}s)")

    all_scores = {}
    best_score = None
    best_parameters = None
    best_results = None
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
