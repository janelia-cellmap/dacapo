from .gp import AddChannelDim, RemoveChannelDim, TransposeDims
import gunpowder as gp
import gunpowder.torch as gp_torch
import zarr


def validate_2d(
        data,
        predictor,
        store_results=None):

    raw_channels = max(1, data.raw.num_channels)
    input_shape = predictor.input_shape
    output_shape = predictor.output_shape
    dataset_shape = data.raw.validate.shape
    dataset_roi = data.raw.validate.roi
    voxel_size = data.raw.validate.voxel_size

    # switch to world units
    input_size = voxel_size*input_shape
    output_size = voxel_size*output_shape

    raw = gp.ArrayKey('RAW')
    gt = gp.ArrayKey('GT')
    target = gp.ArrayKey('TARGET')
    prediction = gp.ArrayKey('PREDICTION')

    channel_dims = 0 if raw_channels == 1 else 1
    data_dims = len(dataset_shape) - channel_dims

    if data_dims == 3:
        num_samples = dataset_shape[0]
        sample_shape = dataset_shape[channel_dims + 1:]
    else:
        raise RuntimeError(
            "For 2D validation, please provide a 3D array where the first "
            "dimension indexes the samples.")

    num_samples = data.raw.validate.num_samples

    sample_shape = gp.Coordinate(sample_shape)
    sample_size = sample_shape*voxel_size

    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(gt, output_size)
    scan_request.add(target, output_size)
    scan_request.add(prediction, output_size)

    # overwrite source ROI to treat samples as z dimension
    spec = gp.ArraySpec(
        roi=gp.Roi(
            (0,) + dataset_roi.get_begin(),
            (num_samples,) + sample_size),
        voxel_size=(1,) + voxel_size)
    sources = (
        data.raw.validate.get_source(raw, overwrite_spec=spec),
        data.gt.validate.get_source(gt, overwrite_spec=spec)
    )
    pipeline = sources + gp.MergeProvider()
    pipeline += gp.Pad(raw, None)
    pipeline += gp.Pad(gt, None)
    # raw: ([c,] s, h, w)
    # gt: ([c,] s, h, w)
    pipeline += gp.Normalize(raw)
    # raw: ([c,] s, h, w)
    # gt: ([c,] s, h, w)
    pipeline += predictor.add_target(gt, target)
    # raw: ([c,] s, h, w)
    # gt: ([c,] s, h, w)
    # target: ([c,] s, h, w)
    if channel_dims == 0:
        pipeline += AddChannelDim(raw)
    if predictor.target_channels == 0:
        pipeline += AddChannelDim(target)
    # raw: (c, s, h, w)
    # gt: ([c,] s, h, w)
    # target: (c, s, h, w)
    pipeline += TransposeDims(raw, (1, 0, 2, 3))
    pipeline += TransposeDims(target, (1, 0, 2, 3))
    # raw: (s, c, h, w)
    # gt: ([c,] s, h, w)
    # target: (s, c, h, w)
    pipeline += gp_torch.Predict(
            model=predictor,
            inputs={'x': raw},
            outputs={0: prediction})
    # raw: (s, c, h, w)
    # gt: ([c,] s, h, w)
    # target: (s, c, h, w)
    # prediction: (s, c, h, w)
    pipeline += gp.Scan(scan_request)

    total_request = gp.BatchRequest()
    total_request.add(raw, sample_size)
    total_request.add(gt, sample_size)
    total_request.add(target, sample_size)
    total_request.add(prediction, sample_size)

    with gp.build(pipeline):
        batch = pipeline.request_batch(total_request)
        scores = predictor.evaluate(
            batch[prediction],
            batch[gt],
            batch[target],
            store_results)
        if store_results:
            f = zarr.open(store_results)
            f['raw'] = batch[raw].data
            f['prediction'] = batch[prediction].data
        return scores


def validate_3d(
        data,
        predictor,
        store_results=None):

    raw_channels = max(1, data.raw.num_channels)
    input_shape = predictor.input_shape
    output_shape = predictor.output_shape
    voxel_size = data.raw.validate.voxel_size

    # switch to world units
    input_size = voxel_size*input_shape
    output_size = voxel_size*output_shape

    raw = gp.ArrayKey('RAW')
    gt = gp.ArrayKey('GT')
    target = gp.ArrayKey('TARGET')
    prediction = gp.ArrayKey('PREDICTION')

    channel_dims = 0 if raw_channels == 1 else 1

    num_samples = data.raw.validate.num_samples
    assert num_samples == 0, (
        "Multiple samples for 3D validation not yet implemented")

    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(gt, output_size)
    scan_request.add(target, output_size)
    scan_request.add(prediction, output_size)

    sources = (
        data.raw.validate.get_source(raw),
        data.gt.validate.get_source(gt))
    pipeline = sources + gp.MergeProvider()
    pipeline += gp.Pad(raw, None)
    pipeline += gp.Pad(gt, None)
    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    pipeline += gp.Normalize(raw)
    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    pipeline += predictor.add_target(gt, target)
    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    # target: ([c,] d, h, w)
    if channel_dims == 0:
        pipeline += AddChannelDim(raw)
    # raw: (c, d, h, w)
    # gt: ([c,] d, h, w)
    # target: ([c,] d, h, w)
    # add a "batch" dimension
    pipeline += AddChannelDim(raw)
    # raw: (1, c, d, h, w)
    # gt: ([c,] d, h, w)
    # target: ([c,] d, h, w)
    pipeline += gp_torch.Predict(
            model=predictor,
            inputs={'x': raw},
            outputs={0: prediction})
    # remove "batch" dimension
    pipeline += RemoveChannelDim(raw)
    pipeline += RemoveChannelDim(prediction)
    # raw: (c, d, h, w)
    # gt: ([c,] d, h, w)
    # target: ([c,] d, h, w)
    # prediction: ([c,] d, h, w)
    if channel_dims == 0:
        pipeline += RemoveChannelDim(raw)
    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    # target: ([c,] d, h, w)
    # prediction: ([c,] d, h, w)
    pipeline += gp.Scan(scan_request)

    # ensure validation ROI is at least the size of the network input
    roi = data.raw.validate.roi.grow(input_size/2, input_size/2)

    total_request = gp.BatchRequest()
    total_request[raw] = gp.ArraySpec(roi=roi)
    total_request[gt] = gp.ArraySpec(roi=roi)
    total_request[target] = gp.ArraySpec(roi=roi)
    total_request[prediction] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(total_request)
        scores = predictor.evaluate(
            batch[prediction],
            batch[gt],
            batch[target],
            store_results)
        if store_results:
            f = zarr.open(store_results)
            f['raw'] = batch[raw].data
            f['prediction'] = batch[prediction].data
        return scores


def validate(
        data,
        predictor,
        store_results=None):

    task_dims = data.raw.spatial_dims

    if task_dims == 2:
        return validate_2d(data, predictor, store_results)
    elif task_dims == 3:
        return validate_3d(data, predictor, store_results)
    else:
        raise RuntimeError(
            "Validation other than 2D/3D not yet implemented")
