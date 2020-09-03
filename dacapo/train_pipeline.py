from .gp import Squash, AddChannelDim, RemoveChannelDim, TransposeDims
import gunpowder as gp
import gunpowder.torch as gp_torch
import math
import os


def create_pipeline_2d(
        task,
        predictor,
        optimizer,
        batch_size,
        outdir,
        snapshot_every):

    raw_channels = task.data.raw.num_channels
    filename = task.data.raw.train.filename
    input_shape = predictor.input_shape
    output_shape = predictor.output_shape
    dataset_shape = task.data.raw.train.shape
    dataset_roi = task.data.raw.train.roi
    voxel_size = task.data.raw.train.voxel_size

    # switch to world units
    input_size = voxel_size*input_shape
    output_size = voxel_size*output_shape

    raw = gp.ArrayKey('RAW')
    gt = gp.ArrayKey('GT')
    target = gp.ArrayKey('TARGET')
    weights = gp.ArrayKey('WEIGHTS')
    prediction = gp.ArrayKey('PREDICTION')

    channel_dims = 0 if raw_channels == 1 else 1
    data_dims = len(dataset_shape) - channel_dims

    if data_dims == 3:
        num_samples = dataset_shape[0]
        sample_shape = dataset_shape[channel_dims + 1:]
    else:
        raise RuntimeError("For 2D training, please provide a 3D array where "
                           "the first dimension indexes the samples.")

    sample_shape = gp.Coordinate(sample_shape)
    sample_size = sample_shape*voxel_size

    # overwrite source ROI to treat samples as z dimension
    spec = gp.ArraySpec(
        roi=gp.Roi(
            (0,) + dataset_roi.get_begin(),
            (num_samples,) + sample_size),
        voxel_size=(1,) + voxel_size)
    sources = (
        task.data.raw.train.get_source(raw, overwrite_spec=spec),
        task.data.gt.train.get_source(gt, overwrite_spec=spec)
    )
    pipeline = sources + gp.MergeProvider()
    pipeline += gp.Pad(raw, None)
    pipeline += gp.Normalize(raw)
    # raw: ([c,] d=1, h, w)
    # gt: ([c,] d=1, h, w)
    pipeline += gp.RandomLocation()
    # raw: ([c,] d=1, h, w)
    # gt: ([c,] d=1, h, w)
    for augmentation in eval(task.augmentations):
        pipeline += augmentation
    pipeline += predictor.add_target(gt, target)
    # (don't care about gt anymore)
    # raw: ([c,] d=1, h, w)
    # target: ([c,] d=1, h, w)
    weights_node = task.loss.add_weights(target, weights)
    if weights_node:
        pipeline += weights_node
        loss_inputs = {0: prediction, 1: target, 2: weights}
    else:
        loss_inputs = {0: prediction, 1: target}
    # raw: ([c,] d=1, h, w)
    # target: ([c,] d=1, h, w)
    # [weights: ([c,] d=1, h, w)]
    # get rid of z dim:
    pipeline += Squash(dim=-3)
    # raw: ([c,] h, w)
    # target: ([c,] h, w)
    # [weights: ([c,] h, w)]
    if channel_dims == 0:
        pipeline += AddChannelDim(raw)
    # raw: (c, h, w)
    # target: ([c,] h, w)
    # [weights: ([c,] h, w)]
    pipeline += gp.PreCache()
    pipeline += gp.Stack(batch_size)
    # raw: (b, c, h, w)
    # target: (b, [c,] h, w)
    # [weights: (b, [c,] h, w)]
    pipeline += gp_torch.Train(
        model=predictor,
        loss=task.loss,
        optimizer=optimizer,
        inputs={'x': raw},
        loss_inputs=loss_inputs,
        outputs={0: prediction},
        save_every=1e6)
    # raw: (b, c, h, w)
    # target: (b, [c,] h, w)
    # [weights: (b, [c,] h, w)]
    # prediction: (b, [c,] h, w)
    if snapshot_every > 0:
        # get channels first
        pipeline += TransposeDims(raw, (1, 0, 2, 3))
        if predictor.target_channels > 0:
            pipeline += TransposeDims(target, (1, 0, 2, 3))
            if weights_node:
                pipeline += TransposeDims(weights, (1, 0, 2, 3))
        if predictor.prediction_channels > 0:
            pipeline += TransposeDims(prediction, (1, 0, 2, 3))
        # raw: (c, b, h, w)
        # target: ([c,] b, h, w)
        # [weights: ([c,] b, h, w)]
        # prediction: ([c,] b, h, w)
        if channel_dims == 0:
            pipeline += RemoveChannelDim(raw)
        # raw: ([c,] b, h, w)
        # target: ([c,] b, h, w)
        # [weights: ([c,] b, h, w)]
        # prediction: ([c,] b, h, w)
        pipeline += gp.Snapshot(
            dataset_names={
                raw: 'raw',
                target: 'target',
                prediction: 'prediction',
                weights: 'weights'
            },
            every=snapshot_every,
            output_dir=os.path.join(outdir, 'snapshots'),
            output_filename="{iteration}.hdf")
    pipeline += gp.PrintProfilingStats(every=100)

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(gt, output_size)
    request.add(target, output_size)
    if weights_node:
        request.add(weights, output_size)
    request.add(prediction, output_size)

    return pipeline, request


def create_pipeline_3d(
        task,
        predictor,
        optimizer,
        batch_size,
        outdir,
        snapshot_every):

    raw_channels = max(1, task.data.raw.num_channels)
    input_shape = predictor.input_shape
    output_shape = predictor.output_shape
    voxel_size = task.data.raw.train.voxel_size

    # switch to world units
    input_size = voxel_size*input_shape
    output_size = voxel_size*output_shape

    raw = gp.ArrayKey('RAW')
    gt = gp.ArrayKey('GT')
    target = gp.ArrayKey('TARGET')
    weights = gp.ArrayKey('WEIGHTS')
    prediction = gp.ArrayKey('PREDICTION')

    channel_dims = 0 if raw_channels == 1 else 1

    num_samples = task.data.raw.train.num_samples
    assert num_samples == 0, (
        "Multiple samples for 3D training not yet implemented")

    sources = (
        task.data.raw.train.get_source(raw),
        task.data.gt.train.get_source(gt))
    pipeline = sources + gp.MergeProvider()
    pipeline += gp.Pad(raw, None)
    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    pipeline += gp.Normalize(raw)
    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    pipeline += gp.RandomLocation()
    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    for augmentation in eval(task.augmentations):
        pipeline += augmentation
    pipeline += predictor.add_target(gt, target)
    # (don't care about gt anymore)
    # raw: ([c,] d, h, w)
    # target: ([c,] d, h, w)
    weights_node = task.loss.add_weights(target, weights)
    if weights_node:
        pipeline += weights_node
        loss_inputs = {0: prediction, 1: target, 2: weights}
    else:
        loss_inputs = {0: prediction, 1: target}
    # raw: ([c,] d, h, w)
    # target: ([c,] d, h, w)
    # [weights: ([c,] d, h, w)]
    if channel_dims == 0:
        pipeline += AddChannelDim(raw)
    # raw: (c, d, h, w)
    # target: ([c,] d, h, w)
    # [weights: ([c,] d, h, w)]
    pipeline += gp.PreCache()
    pipeline += gp.Stack(batch_size)
    # raw: (b, c, d, h, w)
    # target: (b, [c,] d, h, w)
    # [weights: (b, [c,] d, h, w)]
    pipeline += gp_torch.Train(
        model=predictor,
        loss=task.loss,
        optimizer=optimizer,
        inputs={'x': raw},
        loss_inputs=loss_inputs,
        outputs={0: prediction},
        save_every=1e6)
    # raw: (b, c, d, h, w)
    # target: (b, [c,] d, h, w)
    # [weights: (b, [c,] d, h, w)]
    # prediction: (b, [c,] d, h, w)
    if snapshot_every > 0:
        # get channels first
        pipeline += TransposeDims(raw, (1, 0, 2, 3, 4))
        if predictor.target_channels > 0:
            pipeline += TransposeDims(target, (1, 0, 2, 3, 4))
            if weights_node:
                pipeline += TransposeDims(weights, (1, 0, 2, 3, 4))
        if predictor.prediction_channels > 0:
            pipeline += TransposeDims(prediction, (1, 0, 2, 3, 4))
        # raw: (c, b, d, h, w)
        # target: ([c,] b, d, h, w)
        # [weights: ([c,] b, d, h, w)]
        # prediction: ([c,] b, d, h, w)
        if channel_dims == 0:
            pipeline += RemoveChannelDim(raw)
        # raw: ([c,] b, d, h, w)
        # target: (c, b, d, h, w)
        # [weights: ([c,] b, d, h, w)]
        # prediction: (c, b, d, h, w)
        pipeline += gp.Snapshot(
            dataset_names={
                raw: 'raw',
                target: 'target',
                prediction: 'prediction',
                weights: 'weights'
            },
            every=snapshot_every,
            output_dir=os.path.join(outdir, 'snapshots'),
            output_filename="{iteration}.hdf")
    pipeline += gp.PrintProfilingStats(every=100)

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(gt, output_size)
    request.add(target, output_size)
    if weights_node:
        request.add(weights, output_size)
    request.add(prediction, output_size)

    return pipeline, request


def create_train_pipeline(
        task,
        predictor,
        optimizer,
        batch_size,
        outdir='.',
        snapshot_every=1000):

    task_dims = task.data.raw.spatial_dims

    if task_dims == 2:
        return create_pipeline_2d(
            task,
            predictor,
            optimizer,
            batch_size,
            outdir,
            snapshot_every)
    elif task_dims == 3:
        return create_pipeline_3d(
            task,
            predictor,
            optimizer,
            batch_size,
            outdir,
            snapshot_every)
    else:
        raise RuntimeError(
            "Training other than 2D/3D not yet implemented")
