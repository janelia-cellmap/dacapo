from dacapo.experiments.datasplits.keys import ArrayKey
from dacapo.gp import DaCapoArraySource
from funlib.geometry import Coordinate
import daisy
import gunpowder as gp
import gunpowder.torch as gp_torch
import logging
import numpy as np
import zarr

logger = logging.getLogger(__name__)


def predict(model, raw_array, prediction_array, num_cpu_workers=4):
    # get the model's input and output size

    voxel_size = Coordinate(raw_array.voxel_size)
    input_size = voxel_size * Coordinate(model.input_shape)
    output_size = voxel_size * Coordinate(model.output_shape)

    logger.info(
        "Predicting with input size %s, output size %s",
        input_size, output_size)

    # calculate input and output rois

    context = (input_size - output_size) / 2
    input_roi = raw_array.roi
    output_roi = input_roi.grow(-context, -context)

    logger.info(
        "Total input ROI: %s, output ROI: %s",
        input_roi, output_roi)

    # prepare prediction dataset
    daisy.prepare_ds(
        str(prediction_array.container),
        prediction_array.dataset,
        output_roi,
        voxel_size,
        np.float32,
        write_size=output_size,
        num_channels=model.num_out_channels,
        compressor=zarr.storage.default_compressor.get_config())

    # create gunpowder keys

    raw = gp.ArrayKey("RAW")
    prediction = gp.ArrayKey("PREDICTION")

    # assemble prediction pipeline

    # prepare data source
    pipeline = DaCapoArraySource(raw_array, raw)
    # raw: (c, d, h, w)
    pipeline += gp.Pad(raw, Coordinate((None,) * voxel_size.dims))
    # raw: (c, d, h, w)
    pipeline += gp.Unsqueeze([raw])
    # raw: (1, c, d, h, w)

    # predict
    pipeline += gp_torch.Predict(
        model=model,
        inputs={"x": raw},
        outputs={0: prediction},
        array_specs={
            prediction: gp.ArraySpec(
                roi=output_roi,
                voxel_size=voxel_size,
                dtype=np.float32)
        },
        spawn_subprocess=True)
    # raw: (1, c, d, h, w)
    # prediction: (1, [c,] d, h, w)

    # prepare writing
    pipeline += gp.Squeeze([raw, prediction])
    # raw: (c, d, h, w)
    # prediction: (c, d, h, w)
    # raw: (c, d, h, w)
    # prediction: (c, d, h, w)

    # write to zarr
    pipeline += gp.ZarrWrite(
        {prediction: prediction_array.dataset},
        prediction_array.container.parent,
        prediction_array.container.name)

    # create reference batch request
    ref_request = gp.BatchRequest()
    ref_request.add(raw, input_size)
    ref_request.add(prediction, output_size)
    pipeline += gp.Scan(ref_request, num_cpu_workers)

    # build pipeline and predict in complete output ROI

    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())
