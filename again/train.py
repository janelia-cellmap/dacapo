from .datasets import get_dataset_shape, get_dataset_roi, get_voxel_size
from .gp import Squash, AddChannelDim, RemoveChannelDim, TransposeDims
from .prediction_types import Affinities
from .task_types import InstanceSegmentation
import gunpowder as gp
import gunpowder.torch as gp_torch
import os


def create_pipeline_2d(
        task_config,
        model_config,
        optimizer_config,
        model,
        loss,
        optimizer,
        outdir,
        snapshot_every):

    channels = task_config.data.channels
    filename = task_config.data.filename
    input_shape = model_config.input_size
    output_shape = model.output_size(channels, input_shape)
    batch_size = optimizer_config.batch_size
    dataset_shape = get_dataset_shape(filename, 'train/raw')
    dataset_roi = get_dataset_roi(filename, 'train/raw')
    voxel_size = get_voxel_size(filename, 'train/raw')

    # switch to world units
    input_size = voxel_size*input_shape
    output_size = voxel_size*output_shape

    raw = gp.ArrayKey('RAW')
    labels = gp.ArrayKey('LABELS')
    affs = gp.ArrayKey('AFFS')
    weights = gp.ArrayKey('WEIGHTS')
    affs_predicted = gp.ArrayKey('AFFS_PREDICTED')

    channel_dims = 0 if channels == 1 else 1
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
    source_specs = {
        raw: spec,
        labels: spec
    }

    affinity_neighborhood = [(0, 1, 0), (0, 0, 1)]

    pipeline = gp.ZarrSource(
        str(filename),
        {
            raw: 'train/raw',
            labels: 'train/gt'
        },
        array_specs=source_specs)
    pipeline += gp.Normalize(raw)
    # raw: (d=1, h, w)
    # labels: (d=1, h, w)
    pipeline += gp.RandomLocation()
    # raw: (d=1, h, w)
    # labels: (d=1, h, w)
    pipeline += gp.AddAffinities(
        affinity_neighborhood=affinity_neighborhood,
        labels=labels,
        affinities=affs)

    if loss.requires_weights:

        pipeline += gp.BalanceLabels(
            labels=affs,
            scales=weights)
        loss_inputs = {0: affs_predicted, 1: affs, 2: weights}

    else:

        loss_inputs = {0: affs_predicted, 1: affs}

    pipeline += gp.Normalize(affs, factor=1.0)
    # raw: (d=1, h, w)
    # affs: (c=2, d=1, h, w)
    pipeline += Squash(dim=-3)
    # get rid of z dim
    # raw: ([c,] h, w)
    # affs: (c=2, h, w)
    if channel_dims == 0:
        pipeline += AddChannelDim(raw)
    # raw: (c, h, w)
    # affs: (c=2, h, w)
    pipeline += gp.PreCache()
    pipeline += gp.Stack(batch_size)
    # raw: (b, c, h, w)
    # affs: (b, c=2, h, w)
    pipeline += gp_torch.Train(
        model=model,
        loss=loss,
        optimizer=optimizer,
        inputs={'x': raw},
        loss_inputs=loss_inputs,
        outputs={0: affs_predicted},
        save_every=1e6)
    # raw: (b, c, h, w)
    # affs: (b, c=2, h, w)
    # affs_predicted: (b, c=2, h, w)
    if snapshot_every > 0:
        pipeline += TransposeDims(raw, (1, 0, 2, 3))
        pipeline += TransposeDims(affs, (1, 0, 2, 3))
        pipeline += TransposeDims(affs_predicted, (1, 0, 2, 3))
        # raw: (c, b, h, w)
        # affs: (c=2, b, h, w)
        # affs_predicted: (c=2, b, h, w)
        if channel_dims == 0:
            pipeline += RemoveChannelDim(raw)
        # raw: ([c,], b, h, w)
        # affs: (c=2, b, h, w)
        # affs_predicted: (c=2, b, h, w)
        pipeline += gp.Snapshot(
            dataset_names={
                raw: 'raw',
                labels: 'labels',
                affs: 'affs',
                affs_predicted: 'affs_predicted'
            },
            every=snapshot_every,
            output_dir=os.path.join(outdir, 'snapshots'),
            output_filename="{iteration}.hdf")
    pipeline += gp.PrintProfilingStats(every=100)

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(affs, output_size)
    request.add(weights, output_size)
    request.add(affs_predicted, output_size)

    return pipeline, request


def create_pipeline_3d(
        task_config,
        model_config,
        optimizer_config,
        model,
        loss,
        optimizer,
        outdir,
        snapshot_every):

    channels = task_config.data.channels
    filename = task_config.data.filename
    input_shape = model_config.input_size
    output_shape = model.output_size(channels, input_shape)
    batch_size = optimizer_config.batch_size
    dataset_shape = get_dataset_shape(filename, 'train/raw')
    dataset_roi = get_dataset_roi(filename, 'train/raw')
    voxel_size = get_voxel_size(filename, 'train/raw')

    # switch to world units
    input_size = voxel_size*input_shape
    output_size = voxel_size*output_shape

    raw = gp.ArrayKey('RAW')
    labels = gp.ArrayKey('LABELS')
    affs = gp.ArrayKey('AFFS')
    weights = gp.ArrayKey('WEIGHTS')
    affs_predicted = gp.ArrayKey('AFFS_PREDICTED')

    channel_dims = 0 if channels == 1 else 1
    data_dims = len(dataset_shape) - channel_dims

    if data_dims == 3:
        num_samples = 0
        sample_shape = dataset_shape
    elif data_dims == 4:
        num_samples = dataset_shape[0]
        sample_shape = dataset_shape[channel_dims + 1:]
    else:
        raise RuntimeError("For 3D training, please provide a 3D or 4D array "
                           "(for 4D, the first dimension indexes the "
                           "samples).")

    sample_shape = gp.Coordinate(sample_shape)
    sample_size = sample_shape*voxel_size

    if num_samples > 0:
        # overwrite source ROI to treat samples as t dimension
        spec = gp.ArraySpec(
            roi=gp.Roi(
                (0,) + dataset_roi.get_begin(),
                (num_samples,) + sample_size),
            voxel_size=(1,) + voxel_size)
        source_specs = {
            raw: spec,
            labels: spec
        }
    else:
        source_specs = None

    affinity_neighborhood = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    pipeline = gp.ZarrSource(
        str(filename),
        {
            raw: 'train/raw',
            labels: 'train/gt'
        },
        array_specs=source_specs)
    pipeline += gp.Normalize(raw)
    # raw: ([c,] [t=1,], d, h, w)
    # labels: ([t=1,], d, h, w)
    pipeline += gp.RandomLocation()
    # raw: ([c,] [t=1,] d, h, w)
    # labels: ([t=1,] d, h, w)
    if num_samples > 0:
        pipeline += Squash(dim=-4)
    # raw: ([c,] d, h, w)
    # labels: (d, h, w)
    pipeline += gp.AddAffinities(
        affinity_neighborhood=affinity_neighborhood,
        labels=labels,
        affinities=affs)

    if loss.requires_weights:

        pipeline += gp.BalanceLabels(
            labels=affs,
            scales=weights)
        loss_inputs = {0: affs_predicted, 1: affs, 2: weights}

    else:

        loss_inputs = {0: affs_predicted, 1: affs}

    pipeline += gp.Normalize(affs, factor=1.0)
    # raw: ([c,] d, h, w)
    # affs: (c=2, d, h, w)
    if channel_dims == 0:
        pipeline += AddChannelDim(raw)
    # raw: (c, d, h, w)
    # affs: (c=2, d, h, w)
    pipeline += gp.PreCache()
    pipeline += gp.Stack(batch_size)
    # raw: (b, c=1, d, h, w)
    # affs: (b, c=2, d, h, w)
    pipeline += gp_torch.Train(
        model=model,
        loss=loss,
        optimizer=optimizer,
        inputs={'x': raw},
        loss_inputs=loss_inputs,
        outputs={0: affs_predicted},
        save_every=1e6)
    # raw: (b, c=1, d, h, w)
    # affs: (b, c=2, d, h, w)
    # affs_predicted: (b, c=2, d, h, w)
    if snapshot_every > 0:
        # get channels first
        pipeline += TransposeDims(raw, (1, 0, 2, 3, 4))
        pipeline += TransposeDims(affs, (1, 0, 2, 3, 4))
        pipeline += TransposeDims(affs_predicted, (1, 0, 2, 3, 4))
        # raw: (c, b, d, h, w)
        # affs: (c=2, b, d, h, w)
        # affs_predicted: (c=2, b, d, h, w)
        if channel_dims == 0:
            pipeline += RemoveChannelDim(raw)
        # raw: ([c,] b, d, h, w)
        # affs: (c=2, b, d, h, w)
        # affs_predicted: (c=2, b, d, h, w)
        pipeline += gp.Snapshot(
            dataset_names={
                raw: 'raw',
                labels: 'labels',
                affs: 'affs',
                affs_predicted: 'affs_predicted'
            },
            every=snapshot_every,
            output_dir=os.path.join(outdir, 'snapshots'),
            output_filename="{iteration}.hdf")
    pipeline += gp.PrintProfilingStats(every=100)

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(affs, output_size)
    request.add(weights, output_size)
    request.add(affs_predicted, output_size)

    return pipeline, request


def create_train_pipeline(
        task_config,
        model_config,
        optimizer_config,
        model,
        loss,
        optimizer,
        outdir='.',
        snapshot_every=1000):

    have_labels = task_config.type == InstanceSegmentation
    predict_affs = task_config.predict == Affinities
    task_dims = task_config.data.dims

    if not have_labels or not predict_affs:
        raise RuntimeError(
            "Training other than affs from labels not yet implemented")

    if task_dims == 2:
        return create_pipeline_2d(
            task_config,
            model_config,
            optimizer_config,
            model,
            loss,
            optimizer,
            outdir,
            snapshot_every)
    elif task_dims == 3:
        return create_pipeline_3d(
            task_config,
            model_config,
            optimizer_config,
            model,
            loss,
            optimizer,
            outdir,
            snapshot_every)
    else:
        raise RuntimeError(
            "Training other 2D/3D not yet implemented")
