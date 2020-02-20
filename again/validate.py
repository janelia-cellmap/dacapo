from .datasets import get_voxel_size
from .evaluate import evaluate
from .gp import AddChannelDim, RemoveChannelDim, TransposeDims
from .prediction_types import Affinities
from .task_types import InstanceSegmentation
import gunpowder as gp
import gunpowder.torch as gp_torch
import zarr


def validate(task_config, model_config, model, store_results=None):

    have_labels = task_config.type == InstanceSegmentation
    predict_affs = task_config.predict == Affinities
    dims = task_config.data.dims

    if not have_labels or not predict_affs or dims != 2:
        raise RuntimeError(
            "Validation other than 2D affs from labels not yet implemented")

    voxel_size = get_voxel_size(task_config.data.filename, 'validate/raw')
    channels = task_config.data.channels
    filename = task_config.data.filename
    input_shape = gp.Coordinate(model_config.input_size)
    output_shape = gp.Coordinate(model.output_size(channels, input_shape))
    input_size = input_shape*voxel_size
    output_size = output_shape*voxel_size

    dataset_shape = zarr.open(str(filename))['train/raw'].shape
    sample_shape = dataset_shape[1:]
    sample_size = voxel_size*sample_shape

    raw = gp.ArrayKey('RAW')
    labels = gp.ArrayKey('LABELS')
    affs_predicted = gp.ArrayKey('AFFS_PREDICTED')

    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(affs_predicted, output_size)
    scan_request.add(labels, output_size)

    total_request = gp.BatchRequest()
    total_request.add(raw, sample_size)
    total_request.add(affs_predicted, sample_size)
    total_request.add(labels, sample_size)

    pipeline = (
        gp.ZarrSource(
            str(filename),
            {
                raw: 'validate/raw',
                labels: 'validate/gt'
            }) +
        gp.Pad(raw, size=None) +
        gp.Normalize(raw) +
        # raw: (s, h, w)
        # labels: (s, h, w)
        AddChannelDim(raw) +
        # raw: (c=1, s, h, w)
        # labels: (s, h, w)
        TransposeDims(raw, (1, 0, 2, 3)) +
        # raw: (s, c=1, h, w)
        # labels: (s, h, w)
        gp_torch.Predict(
            model=model,
            inputs={'x': raw},
            outputs={0: affs_predicted}) +
        # raw: (s, c=1, h, w)
        # affs_predicted: (s, c=2, h, w)
        # labels: (s, h, w)
        TransposeDims(raw, (1, 0, 2, 3)) +
        RemoveChannelDim(raw) +
        # raw: (s, h, w)
        # affs_predicted: (s, c=2, h, w)
        # labels: (s, h, w)
        gp.PrintProfilingStats(every=100) +
        gp.Scan(scan_request)
    )

    with gp.build(pipeline):
        batch = pipeline.request_batch(total_request)
        scores = evaluate(
            task_config,
            batch[affs_predicted],
            batch[labels],
            store_results)
        if store_results:
            f = zarr.open(store_results)
            f['raw'] = batch[raw].data
            f['affs'] = batch[affs_predicted].data.transpose((1, 0, 2, 3))
        return scores
