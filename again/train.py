from .gp import Squash, AddChannelDim, RemoveChannelDim, TransposeDims
from .prediction_types import Affinities
from .task_types import InstanceSegmentation
import gunpowder as gp
import gunpowder.torch as gp_torch
import os
import torch


def get_output_size(model, channels, input_size):

    # (b=1, c=channels, [d, ] w, h)
    dummy_data = torch.zeros((1, channels) + input_size)
    out = model(dummy_data)
    return out.shape[2:]


def create_train_pipeline(
        task_config,
        optimizer_config,
        model,
        loss,
        optimizer,
        outdir='.',
        snapshot_every=1000):

    have_labels = task_config.type == InstanceSegmentation
    predict_affs = task_config.predict == Affinities
    dims = task_config.data.dims

    if not have_labels or not predict_affs or dims != 2:
        raise RuntimeError(
            "Training other than 2D affs from labels not yet implemented")

    channels = task_config.data.channels
    filename = task_config.data.filename
    input_size = optimizer_config.input_size
    output_size = get_output_size(model, channels, input_size)
    batch_size = optimizer_config.batch_size

    raw = gp.ArrayKey('RAW')
    labels = gp.ArrayKey('LABELS')
    affs = gp.ArrayKey('AFFS')
    affs_predicted = gp.ArrayKey('AFFS_PREDICTED')

    pipeline = (
        gp.ZarrSource(
            str(filename),
            {
                raw: 'train/raw',
                labels: 'train/gt'
            }) +
        # raw: (d=1, h, w)
        # labels: (d=1, h, w)
        gp.RandomLocation() +
        # raw: (d=1, h, w)
        # labels: (d=1, h, w)
        gp.AddAffinities(
            affinity_neighborhood=[(0, 1, 0), (0, 0, 1)],
            labels=labels,
            affinities=affs) +
        gp.Normalize(affs, factor=1.0) +
        # raw: (d=1, h, w)
        # affs: (c=2, d=1, h, w)
        Squash(dim=-3) +
        # get rid of z dim
        # raw: (h, w)
        # affs: (c=2, h, w)
        AddChannelDim(raw) +
        # raw: (c=1, h, w)
        # affs: (c=2, h, w)
        gp.PreCache() +
        gp.Stack(batch_size) +
        # raw: (b=10, c=1, h, w)
        # affs: (b=10, c=2, h, w)
        gp_torch.Train(
            model=model,
            loss=loss,
            optimizer=optimizer,
            inputs={'x': raw},
            target=affs,
            output=affs_predicted) +
        # raw: (b=10, c=1, h, w)
        # affs: (b=10, c=2, h, w)
        # affs_predicted: (b=10, c=2, h, w)
        TransposeDims((1, 0, 2, 3)) +
        # raw: (c=1, b=10, h, w)
        # affs: (c=2, b=10, h, w)
        # affs_predicted: (c=2, b=10, h, w)
        RemoveChannelDim(raw) +
        # raw: (b=10, h, w)
        # affs: (c=2, b=10, h, w)
        # affs_predicted: (c=2, b=10, h, w)
        gp.Snapshot(
            dataset_names={
                raw: 'raw',
                labels: 'labels',
                affs: 'affs',
                affs_predicted: 'affs_predicted'
            },
            every=snapshot_every,
            output_filename=os.path.join(outdir, "{iteration}.hdf")) +
        gp.PrintProfilingStats(every=100)
    )

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(affs, output_size)
    request.add(affs_predicted, output_size)

    return pipeline, request
