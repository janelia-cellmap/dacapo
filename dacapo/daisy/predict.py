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


def predict(
    job_id: str,
    task: Task,
    dataset: Dataset,
    model: Model,
    input_shape: Coordinate,  # Should have config for validation/prediction
    output_shape: Optional[Coordinate],  # Should have config for validation/prediction
    output_dir: Path,
    output_filename: str,
    backbone_checkpoint: Path,
    head_checkpoints: List[Path],
    raw: ArrayDataSource,
    gt: Optional[ArrayDataSource] = None,
    num_workers: int = 5,  # This should be configurable but isn't for validation or prediction
):
    store = MongoDbStore()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = model.instantiate(dataset).to(device)
    backbone.eval()
    heads = [predictor.head(model, dataset).to(device) for predictor in task.predictors]
    [head.eval() for head in heads]

    voxel_size = raw.voxel_size

    # switch to world units
    print(voxel_size, input_shape, output_shape)
    input_size = voxel_size * input_shape
    if output_shape is not None:
        output_size = voxel_size * output_shape
    else:
        output_size = voxel_size * Coordinate(backbone.output_shape(input_shape))

    # calculate input and output rois
    context = (input_size - output_size) / 2
    logger.warning(f"context: {context}")

    input_roi = raw.roi
    output_roi = input_roi.grow(-context, -context)

    # create gunpowder keys
    raw_key = gp.ArrayKey("RAW")
    if gt is not None:
        gt_key = gp.ArrayKey("GT")
    model_output = gp.ArrayKey("MODEL_OUTPUT")

    predictor_keys = {}
    for predictor in task.predictors:
        name = predictor.name
        pred_key = gp.ArrayKey(f"{name.upper()}_PREDICTION")
        target_key = gp.ArrayKey(f"{name.upper()}_TARGET" if gt is not None else None)
        predictor_keys[name] = (pred_key, target_key)

    if gt:
        sources = (
            raw.get_source(raw_key, gp.ArraySpec(interpolatable=True)),
            gt.get_source(gt_key, gp.ArraySpec(interpolatable=False)),
        )
        pipeline = sources + gp.MergeProvider()
    else:
        pipeline = raw.get_source(raw_key, gp.ArraySpec(interpolatable=True))

    pipeline += gp.Pad(raw_key, Coordinate((None,) * voxel_size.dims))
    if gt:
        pipeline += gp.Pad(gt_key, Coordinate((None,) * voxel_size.dims))

    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    pipeline += gp.Normalize(raw_key)
    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    if gt:
        for predictor in task.predictors:
            name = predictor.name
            _, aux_target = predictor_keys[name]
            pipeline += predictor.add_target(gt_key, aux_target, None, None)[0]
    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    # target: ([c,] d, h, w)
    if raw.axes[0] != "c":
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
                roi=output_roi,
                voxel_size=voxel_size,
                dtype=np.float32,
            )
        },
        checkpoint=backbone_checkpoint,
    )
    for head, head_checkpoint, predictor, post_processor in zip(
        heads, head_checkpoints, task.predictors, task.post_processors
    ):
        aux_pred_key, _ = predictor_keys[predictor.name]
        pipeline += gp_torch.Predict(
            model=head,
            inputs={"x": model_output},
            outputs={0: aux_pred_key},
            array_specs={
                aux_pred_key: gp.ArraySpec(
                    roi=output_roi,
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
    if raw.axes[0] != "c":
        pipeline += RemoveChannelDim(raw_key)

    # write generated volumes to zarr
    ds_names = {}

    for predictor in task.predictors:
        name = predictor.name
        pred_key, target_key = predictor_keys[name]
        ds_names[pred_key] = f"volumes/{name}"
        if not Path(f"{output_dir}/{output_filename}/volumes/{name}").exists():
            daisy.prepare_ds(
                f"{output_dir}/{output_filename}",
                f"volumes/{name}",
                output_roi,
                voxel_size,
                np.float32,
                write_size=output_size,
                num_channels=predictor.fmaps_out,
                compressor=zarr.storage.default_compressor.get_config(),
            )
        if gt is not None:
            ds_names[target_key] = f"volumes/{name}_target"
            if not Path(
                f"{output_dir}/{output_filename}/volumes/{name}_target"
            ).exists():
                daisy.prepare_ds(
                    f"{output_dir}/{output_filename}",
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
        output_dir,
        output_filename,
    )

    total_request = gp.BatchRequest()
    total_request[raw_key] = gp.ArraySpec(roi=input_roi)
    total_request[model_output] = gp.ArraySpec(roi=output_roi)
    if gt is not None:
        total_request[gt_key] = gp.ArraySpec(roi=output_roi)
    for aux_name, (pred_key, target_key) in predictor_keys.items():
        total_request[aux_pred_key] = gp.ArraySpec(roi=output_roi)
        if gt:
            total_request[target_key] = gp.ArraySpec(roi=output_roi)

    pred_id = f"{job_id}"
    step_id = "prediction"

    ds_rois = {
        raw_key: "read_roi",
        model_output: "write_roi",
    }
    if gt is not None:
        ds_rois[gt_key] = "write_roi"
    for name, (pred_key, target_key) in predictor_keys.items():
        ds_rois[pred_key] = "write_roi"
        if gt is not None:
            ds_rois[target_key] = "write_roi"

    ref_request = gp.BatchRequest()
    ref_request.add(raw_key, input_size)
    if gt is not None:
        ref_request.add(gt_key, output_size)
    for pred_key, target_key in predictor_keys.values():
        ref_request.add(pred_key, output_size)
        if gt is not None:
            ref_request.add(target_key, output_size)

    # pipeline += gp.DaisyRequestBlocks(
    #     ref_request,
    #     ds_rois,
    #     block_done_callback=lambda block, start, duration: store.mark_block_done(
    #         pred_id, step_id, block.block_id, start, duration
    #     ),
    # )

    def request_block(b):
        request = ref_request.copy()
        for k, v in request.items():
            request[k].roi.shift(b.read_roi.offset)
        with gp.build(pipeline):
            pipeline.request_batch(request)

    daisy_task = daisy.Task(
        task_id="test_1d",
        total_roi=input_roi,
        read_roi=Roi((0,)*input_size.dims, input_size),
        write_roi=Roi(context, output_size),
        process_function=lambda b: request_block(b),
        check_function=lambda b: store.check_block(f"{pred_id}", step_id, b.block_id),
        num_workers=num_workers,
        fit="overhang",
    )
    daisy.run_blockwise([daisy_task])

    # execute the pipeline and write data:
    with gp.build(pipeline):
        pipeline.request_batch(gp.BatchRequest())

    # Return a pipeline that provides desired arrays/graphs.
    sources = {
        "raw": (raw.filename, raw.ds_name),
    }
    if gt:
        sources["gt"] = (gt.filename, gt.ds_name)

    for key, ds_name in ds_names.items():
        sources[str(key).lower()] = (f"{output_dir / output_filename}", ds_name)

    return sources
