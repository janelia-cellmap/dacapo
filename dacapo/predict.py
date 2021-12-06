from dacapo.gp import DaCapoArraySource
from dacapo.compute_context import LocalTorch

from funlib.geometry import Coordinate
import daisy
import gunpowder as gp
import gunpowder.torch as gp_torch
import logging
import numpy as np
import zarr

logger = logging.getLogger(__name__)


def predict(
    model,
    raw_array,
    prediction_array_identifier,
    num_cpu_workers=4,
    compute_context=LocalTorch(),
    output_roi=None,
):
    # get the model's input and output size

    input_voxel_size = Coordinate(raw_array.voxel_size)
    output_voxel_size = model.scale(input_voxel_size)
    input_shape = Coordinate(model.eval_input_shape)
    input_size = input_voxel_size * input_shape
    output_size = output_voxel_size * model.compute_output_shape(input_shape)[1]

    logger.info(
        "Predicting with input size %s, output size %s", input_size, output_size
    )

    # calculate input and output rois

    context = (input_size - output_size) / 2
    if output_roi is None:
        input_roi = raw_array.roi
        output_roi = input_roi.grow(-context, -context)
    else:
        input_roi = output_roi.grow(context, context)

    logger.info("Total input ROI: %s, output ROI: %s", input_roi, output_roi)

    # prepare prediction dataset
    daisy.prepare_ds(
        str(prediction_array_identifier.container),
        prediction_array_identifier.dataset,
        output_roi,
        output_voxel_size,
        np.float32,
        write_size=output_size,
        num_channels=model.num_out_channels,
        compressor={"id": "blosc", "cname": "zstd", "clevel": 6},
    )

    # create gunpowder keys

    raw = gp.ArrayKey("RAW")
    prediction = gp.ArrayKey("PREDICTION")

    # assemble prediction pipeline

    # prepare data source
    pipeline = DaCapoArraySource(raw_array, raw)
    # raw: (c, d, h, w)
    pipeline += gp.Pad(raw, Coordinate((None,) * input_voxel_size.dims))
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
                roi=output_roi, voxel_size=output_voxel_size, dtype=np.float32
            )
        },
        spawn_subprocess=False,
        device=str(compute_context.device),
    )
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
        {prediction: prediction_array_identifier.dataset},
        prediction_array_identifier.container.parent,
        prediction_array_identifier.container.name,
    )

    # create reference batch request
    ref_request = gp.BatchRequest()
    ref_request.add(raw, input_size)
    ref_request.add(prediction, output_size)
    pipeline += gp.Scan(ref_request)

    # build pipeline and predict in complete output ROI

    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())

    container = zarr.open(prediction_array_identifier.container)
    dataset = container[prediction_array_identifier.dataset]
    dataset.attrs["axes"] = (
        raw_array.axes if "c" in raw_array.axes else ["c"] + raw_array.axes
    )
