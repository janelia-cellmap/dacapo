from .predict_pipeline import predict
from dacapo.store import sanatize
import time
import zarr


def validate(
        data,
        model,
        predictor,
        store_best_result=None,
        best_score_name=None,
        best_score_relation=None):

    raw_data = data.raw.validate
    gt_data = data.gt.validate

    print("Predicting on validation data...")
    start = time.time()
    ds = predict(raw_data, model, predictor, gt_data)
    print(f"...done ({time.time() - start}s)")

    if store_best_result:
        f = zarr.open(store_best_result)
        f['raw'] = ds['raw'].data
        f['prediction'] = ds['prediction'].data

    all_scores = {}
    best_score = None
    best_parameters = None
    best_results = None
    for ret in predictor.evaluate(
            ds['prediction'],
            ds['gt'],
            ds['target'],
            store_best_result is not None):

        if store_best_result:
            parameters, scores, results = ret
            score = scores['average'][best_score_name]
            if best_score is None \
                    or best_score_relation(score, best_score) == score:
                best_score = score
                best_parameters = parameters
                best_results = results
        else:
            parameters, scores = ret

        all_scores[str(parameters.id)] = {
            'post_processing_parameters': parameters.to_dict(),
            'scores': scores
        }

    if store_best_result:
        f = zarr.open(store_best_result)
        for k, v in best_results.items():
            f[k] = v
        d = sanatize(best_parameters.to_dict())
        for k, v in d.items():
            f.attrs[k] = v

    return all_scores
