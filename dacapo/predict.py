# import torch
from dacapo.gp import DaCapoArraySource
from dacapo.experiments.model import Model
from dacapo.experiments.datasplits.datasets.arrays import Array
from dacapo.store.local_array_store import LocalArrayIdentifier
from dacapo.compute_context import LocalTorch, ComputeContext
from dacapo.experiments.datasplits.datasets.arrays.zarr_array import ZarrArray

from funlib.geometry import Coordinate, Roi
import gunpowder as gp
import gunpowder.torch as gp_torch
import numpy as np
import zarr

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def predict(
    model: Model,
    raw_array: Array,
    prediction_array_identifier: LocalArrayIdentifier,
    num_cpu_workers: int = 4,
    # compute_context: ComputeContext = LocalTorch(),
    output_roi: Optional[Roi] = None,
):
    # get the model's input and output size

    input_voxel_size = Coordinate(raw_array.voxel_size)
    output_voxel_size = model.scale(input_voxel_size)
    input_shape = Coordinate(model.eval_input_shape)
    input_size = input_voxel_size * input_shape
    output_size = output_voxel_size * model.compute_output_shape(input_shape)[1]

    # check if model not in cuda move it to cuda
    # if not model.is_cuda:
    #     model = model.eval().to("cuda")

    # if not torch.cuda.is_available():
    #     raise ValueError("CUDA is not available")
    
    # if not(next(model.parameters()).is_cuda):
    #     raise ValueError("Model is not on CUDA")
    

    logger.warning(
        "Predicting with input size %s, output size %s", input_size, output_size
    )

    # calculate input and output rois

    context = (input_size - output_size) / 2
    if output_roi is None:
        input_roi = raw_array.roi
        output_roi = input_roi.grow(-context, -context)
    else:
        input_roi = output_roi.grow(context, context)

    logger.warning("Total input ROI: %s, output ROI: %s", input_roi, output_roi)

    # prepare prediction dataset
    axes = ["c"] + [axis for axis in raw_array.axes if axis != "c"]
    import os 
    dataset_path = os.path.join(prediction_array_identifier.container, prediction_array_identifier.dataset)
    if os.path.exists(dataset_path):
        logger.warning(f"Removing existing dataset at {dataset_path}")
        import shutil
        shutil.rmtree(dataset_path)
    ZarrArray.create_from_array_identifier(
        prediction_array_identifier,
        axes,
        output_roi,
        model.num_out_channels,
        output_voxel_size,
        np.float32,
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

    gt_padding = (output_size - output_roi.shape) % output_size
    prediction_roi = output_roi.grow(gt_padding)

    # predict
    pipeline += gp_torch.Predict(
        model=model,
        inputs={"x": raw},
        outputs={0: prediction},
        array_specs={
            prediction: gp.ArraySpec(
                roi=prediction_roi, voxel_size=output_voxel_size, dtype=np.float32
            )
        },
        spawn_subprocess=False,
        device="cuda",
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
        dataset_dtypes={prediction: np.float32},
    )

    # create reference batch request
    ref_request = gp.BatchRequest()
    ref_request.add(raw, input_size)
    ref_request.add(prediction, output_size)
    pipeline += gp.Scan(ref_request)

    # build pipeline and predict in complete output ROI

    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())
    
    logger.warning("Finished predicting")

    container = zarr.open(prediction_array_identifier.container)
    dataset = container[prediction_array_identifier.dataset]
    dataset.attrs["axes"] = (
        raw_array.axes if "c" in raw_array.axes else ["c"] + raw_array.axes
    )

    logger.warning("Finished writing to %s", prediction_array_identifier)