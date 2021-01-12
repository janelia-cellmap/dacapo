from .gp import AddChannelDim, RemoveChannelDim
from .padding import compute_padding

import gunpowder as gp
import gunpowder.torch as gp_torch
import daisy

import numpy as np

from dacapo.store import MongoDbStore

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def predict_pipeline(
    raw,
    model,
    predictor,
    output_dir,
    output_filename,
    run_hash=None,
    gt=None,
    aux_tasks=None,
    total_roi=None,
    model_padding=None,
    daisy_worker=False,
    checkpoint=None,
    padding_mode="same",
):
    """
    store: a mongodb collection in which to store completed blocks
    """

    raw_channels = max(1, raw.num_channels)
    if model_padding is not None:
        input_shape = tuple(a + b for a, b in zip(model.input_shape, model_padding))
        output_shape = tuple(a + b for a, b in zip(model.output_shape, model_padding))
    else:
        input_shape = model.input_shape
        output_shape = model.output_shape

    voxel_size = raw.voxel_size

    # switch to world units
    input_size = voxel_size * input_shape
    output_size = voxel_size * output_shape

    # calculate input and output rois
    context = (input_size - output_size) / 2
    logger.warning(f"context: {context}")

    input_roi = raw.roi
    if gt is not None:
        output_roi = gt.roi
        if total_roi is not None:
            output_roi = output_roi.intersect(gp.Roi(*total_roi))
    else:
        output_roi = input_roi
        if total_roi is not None:
            output_roi = input_roi.intersect(gp.Roi(*total_roi))

    input_roi, output_roi, padding = compute_padding(
        input_roi,
        output_roi,
        input_size,
        output_size,
        voxel_size,
        padding=padding_mode,
    )

    # create gunpowder keys
    raw_key = gp.ArrayKey("RAW")
    gt_key = gp.ArrayKey("GT")
    target = gp.ArrayKey("TARGET")
    model_output = gp.ArrayKey("MODEL_OUTPUT")
    prediction = gp.ArrayKey("PREDICTION")

    aux_keys = {}
    if aux_tasks is not None:
        for name, _, _ in aux_tasks:
            aux_keys[name] = (
                gp.ArrayKey(f"{name.upper()}_PREDICTION"),
                gp.ArrayKey(f"{name.upper()}_TARGET"),
            )

    channel_dims = 0 if raw_channels == 1 else 1

    num_samples = raw.num_samples
    assert num_samples == 0, "Multiple samples for 3D validation not yet implemented"

    if gt is not None:
        target_node, extra_gt_padding = predictor.add_target(gt_key, target)
        if extra_gt_padding is None:
            extra_gt_padding = gp.Coordinate((0,) * len(padding))
        for aux_name, aux_predictor, _ in aux_tasks:
            _, aux_target = aux_keys[name]
            aux_extra_gt_padding = aux_predictor.add_target(gt_key, aux_target)[1]
            if aux_extra_gt_padding is not None:
                extra_gt_padding = gp.Coordinate(
                    tuple(
                        max(a, b)
                        for a, b in zip(aux_extra_gt_padding, extra_gt_padding)
                    )
                )
    else:
        extra_gt_padding = gp.Coordinate((0,) * len(padding))

    padding_in_voxel_fractions = np.asarray(padding, dtype=np.float32) / np.asarray(
        voxel_size
    )
    extra_in_voxel_fractions = np.asarray(
        extra_gt_padding, dtype=np.float32
    ) / np.asarray(voxel_size)

    padding = gp.Coordinate(np.ceil(padding_in_voxel_fractions))
    extra_gt_padding = gp.Coordinate(np.ceil(extra_in_voxel_fractions))

    if gt:
        sources = (
            raw.get_source(raw_key, gp.ArraySpec(interpolatable=True)),
            gt.get_source(gt_key, gp.ArraySpec(interpolatable=False)),
        )
        pipeline = sources + gp.MergeProvider()
    else:
        pipeline = raw.get_source(raw_key, gp.ArraySpec(interpolatable=True))

    pipeline += gp.Pad(raw_key, padding)
    if gt:
        pipeline += gp.Pad(gt_key, padding + extra_gt_padding)

    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    pipeline += gp.Normalize(raw_key)
    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    if gt:
        pipeline += target_node
        if aux_tasks is not None:
            for i, (aux_name, aux_predictor, _) in enumerate(aux_tasks):
                _, aux_target = aux_keys[name]
                pipeline += aux_predictor.add_target(gt_key, aux_target)[0]
    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    # target: ([c,] d, h, w)
    if channel_dims == 0:
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
        model=model,
        inputs={"x": raw_key},
        outputs={0: model_output},
        array_specs={
            prediction: gp.ArraySpec(
                roi=output_roi,
                voxel_size=voxel_size,
                dtype=np.float32,
            )
        },
        checkpoint=checkpoint,
        state_dict_key="model_state_dict",
    )
    pipeline += gp_torch.Predict(
        model=predictor,
        inputs={"x": model_output},
        outputs={0: prediction},
        array_specs={
            prediction: gp.ArraySpec(
                roi=output_roi,
                voxel_size=voxel_size,
                dtype=np.float32,
            )
        },
        checkpoint=checkpoint,
        state_dict_key="main_0_state_dict",
    )
    if aux_tasks is not None:
        for i, (aux_name, aux_predictor, _) in enumerate(aux_tasks):
            aux_pred_key, _ = aux_keys[name]
            pipeline += gp_torch.Predict(
                model=aux_predictor,
                inputs={"x": model_output},
                outputs={0: aux_pred_key},
                array_specs={
                    aux_pred_key: gp.ArraySpec(
                        roi=output_roi,
                        voxel_size=voxel_size,
                        dtype=np.float32,
                    )
                },
                checkpoint=checkpoint,
                state_dict_key=f"{aux_name}_{i+1}_state_dict",
            )
            pipeline += RemoveChannelDim(aux_pred_key)
    # remove "batch" dimension
    pipeline += RemoveChannelDim(raw_key)
    pipeline += RemoveChannelDim(prediction)
    # raw: (c, d, h, w)
    # gt: ([c,] d, h, w)
    # target: ([c,] d, h, w)
    # prediction: ([c,] d, h, w)
    if channel_dims == 0:
        pipeline += RemoveChannelDim(raw_key)

    scan_request = gp.BatchRequest()
    scan_request.add(raw_key, input_size)
    scan_request.add(model_output, output_size)
    scan_request.add(prediction, output_size)
    for aux_name, aux_key in aux_predictions:
        scan_request.add(aux_key, output_size)
    if gt:
        scan_request.add(gt_key, output_size)
        scan_request.add(target, output_size)

    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    # target: ([c,] d, h, w)
    # prediction: ([c,] d, h, w)
    pipeline += gp.Scan(scan_request)

    # write generated volumes to zarr
    ds_names = {prediction: "volumes/prediction"}
    if not Path(f"{output_dir}/{output_filename}/volumes/prediction").exists():
        daisy.prepare_ds(
            f"{output_dir}/{output_filename}",
            "volumes/prediction",
            daisy.Roi(output_roi.get_offset(), output_roi.get_shape()),
            voxel_size,
            np.float32,
            write_size=output_size,
            num_channels=predictor.output_channels,
        )
    if gt:
        ds_names[target] = "volumes/target"
        if not Path(f"{output_dir}/{output_filename}/volumes/target").exists():
            daisy.prepare_ds(
                f"{output_dir}/{output_filename}",
                "volumes/target",
                daisy.Roi(output_roi.get_offset(), output_roi.get_shape()),
                voxel_size,
                np.float32,
                write_size=output_size,
                num_channels=predictor.output_channels,
            )
    if aux_tasks is not None:
        for aux_name, aux_predictor, _ in aux_tasks:
            aux_pred_key = gp.ArrayKey(f"PRED_{aux_name.upper()}")
            ds_names[aux_pred_key] = f"volumes/{aux_name}"
            if not Path(f"{output_dir}/{output_filename}/volumes/{aux_name}").exists():
                daisy.prepare_ds(
                    f"{output_dir}/{output_filename}",
                    f"volumes/{aux_name}",
                    daisy.Roi(output_roi.get_offset(), output_roi.get_shape()),
                    voxel_size,
                    np.float32,
                    write_size=output_size,
                    num_channels=aux_predictor.output_channels,
                )
    pipeline += gp.ZarrWrite(
        ds_names,
        output_dir,
        output_filename,
    )

    total_request = gp.BatchRequest()
    total_request[raw_key] = gp.ArraySpec(roi=input_roi)
    # total_request[model_output] = gp.ArraySpec(roi=output_roi)
    total_request[prediction] = gp.ArraySpec(roi=output_roi)
    for aux_name, aux_key in aux_predictions:
        total_request[aux_key] = gp.ArraySpec(roi=output_roi)
    if gt:
        total_request[gt_key] = gp.ArraySpec(roi=output_roi)
        total_request[target] = gp.ArraySpec(roi=output_roi)

    # If using daisy, add the daisy block manager.
    if daisy_worker:

        pred_id = f"{run_hash}_predict"
        step_id = "prediction"
        store = MongoDbStore()

        ref_request = scan_request.copy()
        ds_rois = {
            raw_key: "read_roi",
            prediction: "write_roi",
            model_output: "write_roi",
        }
        for aux_name, aux_key in aux_predictions:
            ds_rois[aux_key] = "write_roi"
        if gt:
            ds_rois[gt] = "write_roi"
            ds_rois[target] = "write_roi"
        pipeline += gp.DaisyRequestBlocks(
            ref_request,
            ds_rois,
            block_done_callback=lambda block, start, duration: store.mark_block_done(
                pred_id, step_id, block.block_id, start, duration
            ),
        )
        total_request = gp.BatchRequest()

    # Return a pipeline that provides desired arrays/graphs.
    if gt:
        sources = {
            "raw": (raw.filename, raw.ds_name),
            "gt": (gt.filename, gt.ds_name),
        }
        for key, ds_name in ds_names.items():
            print(f"Opening dataset: {output_dir / output_filename}, {ds_name}")
            sources[str(key).lower()] = (f"{output_dir / output_filename}", ds_name)
    else:
        sources = {
            "raw": (raw.filename, raw.ds_name),
        }
        for key, ds_name in ds_names.items():
            sources[str(key).lower()] = (
                f"{Path(output_dir, output_filename)}",
                ds_name,
            )

    return pipeline, sources, total_request


def predict(
    run_hash,
    raw,
    model,
    predictor,
    output_dir,
    output_filename,
    gt=None,
    aux_tasks=None,
    total_roi=None,
    model_padding=None,
    daisy_worker=False,
):
    model.eval()
    predictor.eval()
    if aux_tasks is not None:
        for aux_name, aux_predictor, _ in aux_tasks:
            aux_predictor.eval()

    compute_pipeline, sources, total_request = predict_pipeline(
        run_hash,
        raw,
        model,
        predictor,
        output_dir,
        output_filename,
        gt=gt,
        aux_tasks=aux_tasks,
        total_roi=total_roi,
        model_padding=model_padding,
        daisy_worker=daisy_worker,
    )

    with gp.build(compute_pipeline):
        compute_pipeline.request_batch(total_request)

    model.train()
    predictor.train()
    if aux_tasks is not None:
        for aux_name, aux_predictor, _ in aux_tasks:
            aux_predictor.train()

    sources = {
        k: daisy.open_ds(filename, ds_name)
        for k, (filename, ds_name) in sources.items()
    }

    return sources
