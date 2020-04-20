from .predict import predict
import zarr


def validate(
        data,
        predictor,
        store_results=None):

    raw_data = data.raw.validate
    gt_data = data.gt.validate

    ds = predict(raw_data, predictor, gt_data)

    if store_results:
        f = zarr.open(store_results)
        f['raw'] = ds['raw'].data
        f['prediction'] = ds['prediction'].data

    return predictor.evaluate(
        ds['prediction'],
        ds['gt'],
        ds['target'],
        store_results)
