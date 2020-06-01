from .gp import AddChannelDim, RemoveChannelDim, TransposeDims
import gunpowder as gp
import gunpowder.torch as gp_torch


def predict_2d(
        raw_data,
        gt_data,
        predictor):

    raw_channels = max(1, raw_data.num_channels)
    input_shape = predictor.input_shape
    output_shape = predictor.output_shape
    dataset_shape = raw_data.shape
    dataset_roi = raw_data.roi
    voxel_size = raw_data.voxel_size

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

    num_samples = raw_data.num_samples

    sample_shape = gp.Coordinate(sample_shape)
    sample_size = sample_shape*voxel_size

    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(prediction, output_size)
    if gt_data:
        scan_request.add(gt, output_size)
        scan_request.add(target, output_size)

    # overwrite source ROI to treat samples as z dimension
    spec = gp.ArraySpec(
        roi=gp.Roi(
            (0,) + dataset_roi.get_begin(),
            (num_samples,) + sample_size),
        voxel_size=(1,) + voxel_size)
    if gt_data:
        sources = (
            raw_data.get_source(raw, overwrite_spec=spec),
            gt_data.get_source(gt, overwrite_spec=spec)
        )
        pipeline = sources + gp.MergeProvider()
    else:
        pipeline = raw_data.get_source(raw, overwrite_spec=spec)
    pipeline += gp.Pad(raw, None)
    if gt_data:
        pipeline += gp.Pad(gt, None)
    # raw: ([c,] s, h, w)
    # gt: ([c,] s, h, w)
    pipeline += gp.Normalize(raw)
    # raw: ([c,] s, h, w)
    # gt: ([c,] s, h, w)
    if gt_data:
        pipeline += predictor.add_target(gt, target)
    # raw: ([c,] s, h, w)
    # gt: ([c,] s, h, w)
    # target: ([c,] s, h, w)
    if channel_dims == 0:
        pipeline += AddChannelDim(raw)
    if gt_data and predictor.target_channels == 0:
        pipeline += AddChannelDim(target)
    # raw: (c, s, h, w)
    # gt: ([c,] s, h, w)
    # target: (c, s, h, w)
    pipeline += TransposeDims(raw, (1, 0, 2, 3))
    if gt_data:
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
    total_request.add(prediction, sample_size)
    if gt_data:
        total_request.add(gt, sample_size)
        total_request.add(target, sample_size)

    with gp.build(pipeline):
        batch = pipeline.request_batch(total_request)
        ret = {
            'raw': batch[raw],
            'prediction': batch[prediction]
        }
        if gt_data:
            ret.update({
                'gt': batch[gt],
                'target': batch[target]
            })
        return ret


def predict_3d(
        raw_data,
        gt_data,
        predictor):

    raw_channels = max(1, raw_data.num_channels)
    input_shape = predictor.input_shape
    output_shape = predictor.output_shape
    voxel_size = raw_data.voxel_size

    # switch to world units
    input_size = voxel_size*input_shape
    output_size = voxel_size*output_shape

    raw = gp.ArrayKey('RAW')
    gt = gp.ArrayKey('GT')
    target = gp.ArrayKey('TARGET')
    prediction = gp.ArrayKey('PREDICTION')

    channel_dims = 0 if raw_channels == 1 else 1

    num_samples = raw_data.num_samples
    assert num_samples == 0, (
        "Multiple samples for 3D validation not yet implemented")

    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(prediction, output_size)
    if gt_data:
        scan_request.add(gt, output_size)
        scan_request.add(target, output_size)

    if gt_data:
        sources = (
            raw_data.get_source(raw),
            gt_data.get_source(gt))
        pipeline = sources + gp.MergeProvider()
    else:
        pipeline = raw_data.get_source(raw)
    pipeline += gp.Pad(raw, None)
    if gt_data:
        pipeline += gp.Pad(gt, None)
    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    pipeline += gp.Normalize(raw)
    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    if gt_data:
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
    roi = raw_data.roi.grow(input_size/2, input_size/2)

    total_request = gp.BatchRequest()
    total_request[raw] = gp.ArraySpec(roi=roi)
    total_request[prediction] = gp.ArraySpec(roi=roi)
    if gt_data:
        total_request[gt] = gp.ArraySpec(roi=roi)
        total_request[target] = gp.ArraySpec(roi=roi)

    with gp.build(pipeline):
        batch = pipeline.request_batch(total_request)
        ret = {
            'raw': batch[raw],
            'prediction': batch[prediction]
        }
        if gt_data:
            ret.update({
                'gt': batch[gt],
                'target': batch[target]
            })
        return ret


def predict(
        raw,
        predictor,
        gt=None):

    task_dims = raw.spatial_dims

    if task_dims == 2:
        return predict_2d(raw, gt, predictor)
    elif task_dims == 3:
        return predict_3d(raw, gt, predictor)
    else:
        raise RuntimeError(
            "Validation other than 2D/3D not yet implemented")
