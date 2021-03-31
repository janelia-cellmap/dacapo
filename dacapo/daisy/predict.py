from dacapo.tasks import Task
from dacapo.data import Dataset, ArrayDataSource
from dacapo.models import Model

from funlib.geometry import Coordinate, Roi


from dacapo.gp import AddChannelDim, RemoveChannelDim
from dacapo.store import MongoDbStore

import gunpowder as gp
import gunpowder.torch as gp_torch
import daisy
import zarr
import torch

import numpy as np

import logging
from pathlib import Path
from typing import Optional, List


logger = logging.getLogger(__name__)

# debug file handler to print out all logs
"""
fh = logging.FileHandler("spam.log")
fh.setLevel(logging.DEBUG)

logger.addHandler(fh)
"""

def predict(
    predict_config,
):
    daisy.Client()
    store = MongoDbStore()
    run = store.get_run(predict_config.run_id)
    task = store.get_task(run.task)
    dataset = store.get_dataset(run.dataset)
    model = store.get_model(run.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = model.instantiate(dataset).to(device)
    backbone.eval()
    heads = [predictor.head(model, dataset).to(device) for predictor in task.predictors]
    [head.eval() for head in heads]

    voxel_size = predict_config.raw.voxel_size

    # switch to world units
    input_size = voxel_size * predict_config.input_shape
    if predict_config.output_shape is not None:
        output_size = voxel_size * predict_config.output_shape
    else:
        output_size = voxel_size * Coordinate(
            backbone.output_shape(predict_config.input_shape)
        )

    # calculate input and output rois
    context = (input_size - output_size) / 2
    logger.warning(f"context: {context}")

    input_roi = predict_config.raw.roi
    output_roi = input_roi.grow(-context, -context)

    # create gunpowder keys
    raw_key = gp.ArrayKey("RAW")
    if predict_config.gt is not None:
        gt_key = gp.ArrayKey("GT")
    model_output = gp.ArrayKey("MODEL_OUTPUT")

    predictor_keys = {}
    for predictor in task.predictors:
        name = predictor.name
        pred_key = gp.ArrayKey(f"{name.upper()}_PREDICTION")
        target_key = gp.ArrayKey(
            f"{name.upper()}_TARGET" if predict_config.gt is not None else None
        )
        predictor_keys[name] = (pred_key, target_key)

    if predict_config.gt:
        sources = (
            predict_config.raw.get_source(raw_key, gp.ArraySpec(interpolatable=True)),
            predict_config.gt.get_source(gt_key, gp.ArraySpec(interpolatable=False)),
        )
        pipeline = sources + gp.MergeProvider()
    else:
        pipeline = predict_config.raw.get_source(
            raw_key, gp.ArraySpec(interpolatable=True)
        )

    pipeline += gp.Pad(raw_key, Coordinate((None,) * voxel_size.dims))
    if predict_config.gt:
        pipeline += gp.Pad(gt_key, Coordinate((None,) * voxel_size.dims))

    with gp.build(pipeline):
        # pipeline provides an infinite roi
        provided_roi = pipeline.spec[raw_key].roi

    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    pipeline += gp.Normalize(raw_key)
    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    if predict_config.gt:
        for predictor in task.predictors:
            name = predictor.name
            _, aux_target = predictor_keys[name]
            pipeline += predictor.add_target(gt_key, aux_target, None, None)[0]
    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    # target: ([c,] d, h, w)
    if predict_config.raw.axes[0] != "c":
        pipeline += AddChannelDim(raw_key)
    # raw: (c, d, h, w)
    # gt: ([c,] d, h, w)
    # target: ([c,] d, h, w)
    # add a "batch" dimension
    pipeline += AddChannelDim(raw_key)
    # raw: (1, c, d, h, w)
    # gt: ([c,] d, h, w)
    # target: ([c,] d, h, w)
    pipeline += gp_torch.Predict(
        model=backbone,
        inputs={"x": raw_key},
        outputs={0: model_output},
        array_specs={
            model_output: gp.ArraySpec(
                roi=provided_roi,
                voxel_size=voxel_size,
                dtype=np.float32,
            )
        },
        checkpoint=predict_config.backbone_checkpoint,
    )
    for head, head_checkpoint, predictor, post_processor in zip(
        heads, predict_config.head_checkpoints, task.predictors, task.post_processors
    ):
        aux_pred_key, _ = predictor_keys[predictor.name]
        pipeline += gp_torch.Predict(
            model=head,
            inputs={"x": model_output},
            outputs={0: aux_pred_key},
            array_specs={
                aux_pred_key: gp.ArraySpec(
                    roi=provided_roi,
                    voxel_size=voxel_size,
                    dtype=np.float32,
                )
            },
            checkpoint=head_checkpoint,
        )
        pipeline += RemoveChannelDim(aux_pred_key)
    # remove "batch" dimension
    pipeline += RemoveChannelDim(raw_key)
    pipeline += RemoveChannelDim(model_output)
    # raw: (c, d, h, w)
    # gt: ([c,] d, h, w)
    # target: ([c,] d, h, w)
    # prediction: ([c,] d, h, w)
    if predict_config.raw.axes[0] != "c":
        pipeline += RemoveChannelDim(raw_key)

    # write generated volumes to zarr
    ds_names = {}

    for predictor in task.predictors:
        name = predictor.name
        pred_key, target_key = predictor_keys[name]
        ds_names[pred_key] = f"volumes/{name}"
        if not Path(
            f"{predict_config.out_container}/volumes/{name}"
        ).exists():
            daisy.prepare_ds(
                f"{predict_config.out_container}",
                f"volumes/{name}",
                output_roi,
                voxel_size,
                np.float32,
                write_size=output_size,
                num_channels=predictor.fmaps_out,
                compressor=zarr.storage.default_compressor.get_config(),
            )
        if predict_config.gt is not None:
            ds_names[target_key] = f"volumes/{name}_target"
            if not Path(
                f"{predict_config.out_container}"
                f"/volumes/{name}_target"
            ).exists():
                daisy.prepare_ds(
                    f"{predict_config.out_container}",
                    f"volumes/{name}_target",
                    output_roi,
                    voxel_size,
                    np.float32,
                    write_size=output_size,
                    num_channels=predictor.fmaps_out,
                    compressor=zarr.storage.default_compressor.get_config(),
                )
    pipeline += gp.ZarrWrite(
        ds_names,
        f"{predict_config.output_dir}",
        f"{predict_config.output_filename}",
    )

    ds_rois = {
        raw_key: "read_roi",
        model_output: "write_roi",
    }
    if predict_config.gt is not None:
        ds_rois[gt_key] = "write_roi"
    for name, (pred_key, target_key) in predictor_keys.items():
        ds_rois[pred_key] = "write_roi"
        if predict_config.gt is not None:
            ds_rois[target_key] = "write_roi"

    ref_request = gp.BatchRequest()
    ref_request.add(raw_key, input_size)
    if predict_config.gt is not None:
        ref_request.add(gt_key, output_size)
    for pred_key, target_key in predictor_keys.values():
        ref_request.add(pred_key, output_size)
        if predict_config.gt is not None:
            ref_request.add(target_key, output_size)

    step_id = "prediction"

    pipeline += gp.DaisyRequestBlocks(
        ref_request,
        ds_rois,
        # block_done_callback=lambda block, start, duration: store.mark_block_done(
        #     predict_config.task_id, step_id, block.block_id, start, duration
        # ),
    )
    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())

    # # Return a pipeline that provides desired arrays/graphs.
    # sources = {
    #     "raw": (raw.filename, raw.ds_name),
    # }
    # if gt:
    #     sources["gt"] = (gt.filename, gt.ds_name)

    # for key, ds_name in ds_names.items():
    #     sources[str(key).lower()] = (f"{output_dir / output_filename}", ds_name)

    # return sources
