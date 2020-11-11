from .gp import AddChannelDim, RemoveChannelDim
import gunpowder as gp
import gunpowder.torch as gp_torch

import logging

logger = logging.getLogger(__name__)


def predict_pipeline(
    raw,
    model,
    predictor,
    output_dir,
    output_filename,
    gt=None,
    aux_tasks=None,
    total_roi=None,
    model_padding=None,
    daisy=False,
):
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

    raw_key = gp.ArrayKey("RAW")
    gt_key = gp.ArrayKey("GT")
    target = gp.ArrayKey("TARGET")
    model_output = gp.ArrayKey("MODEL_OUTPUT")
    prediction = gp.ArrayKey("PREDICTION")

    channel_dims = 0 if raw_channels == 1 else 1

    num_samples = raw.num_samples
    assert num_samples == 0, "Multiple samples for 3D validation not yet implemented"

    if gt:
        sources = (raw.get_source(raw_key), gt.get_source(gt_key))
        pipeline = sources + gp.MergeProvider()
    else:
        pipeline = raw.get_source(raw_key)
    pipeline += gp.Pad(raw_key, None)
    if gt:
        pipeline += gp.Pad(gt_key, None)
    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    pipeline += gp.Normalize(raw_key)
    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    if gt:
        pipeline += predictor.add_target(gt_key, target)
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
        model=model, inputs={"x": raw_key}, outputs={0: model_output}
    )
    pipeline += gp_torch.Predict(
        model=predictor, inputs={"x": model_output}, outputs={0: prediction}
    )
    aux_predictions = []
    for aux_name, aux_predictor, _ in aux_tasks:
        aux_pred_key = gp.ArrayKey(f"PRED_{aux_name.upper()}")
        pipeline += gp_torch.Predict(
            model=aux_predictor, inputs={"x": model_output}, outputs={0: aux_pred_key}
        )
        aux_predictions.append((aux_name, aux_pred_key))
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

    ds_names = {prediction: "volumes/prediction"}
    if gt:
        ds_names[target] = "volumes/target"
    pipeline += gp.ZarrWrite(
        ds_names,
        output_dir,
        output_filename,
    )

    # only output where the gt exists
    context = (input_size - output_size) / 2

    raw_roi = raw.roi
    if total_roi is not None:
        raw_roi = raw_roi.intersect(gp.Roi(*total_roi))
    raw_output_roi = raw_roi.grow(-context, -context)
    if gt is not None:
        output_roi = gt.roi.intersect(raw.roi.grow(-context, -context))
    else:
        output_roi = raw_output_roi
    input_roi = output_roi.grow(context, context)

    assert all([a > b for a, b in zip(input_roi.get_shape(), input_size)])
    assert all([a > b for a, b in zip(output_roi.get_shape(), output_size)])

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
    if daisy:
        ref_request = scan_request.copy()
        ds_rois = {raw: "read_roi", prediction: "write_roi"}
        for aux_name, aux_key in aux_predictions:
            ds_rois[aux_key] = "write_roi"
        if gt:
            ds_rois[gt] = "write_roi"
            ds_rois[target] = "write_roi"
        pipeline += gp.DaisyRequestBlocks(ref_request, ds_rois)
        total_request = gp.BatchRequest()

    # Return a pipeline that provides desired arrays/graphs.
    if gt:
        sources = (
            raw.get_source(raw_key),
            gt.get_source(gt_key),
            gp.ZarrSource(output_filename, ds_names),
        ) + gp.MergeProvider()
    else:
        sources = (
            raw.get_source(raw_key),
            gp.ZarrSource(output_filename, ds_names),
        ) + gp.MergeProvider()

    return pipeline, sources, total_request


def predict(
    raw,
    model,
    predictor,
    output_dir,
    output_filename,
    gt=None,
    aux_tasks=None,
    total_roi=None,
    model_padding=None,
    daisy=False,
):

    compute_pipeline, source_pipeline, total_request = predict_pipeline(
        raw,
        model,
        predictor,
        output_dir,
        output_filename,
        gt=None,
        aux_tasks=None,
        total_roi=None,
        model_padding=None,
        daisy=False,
    )

    with gp.build(compute_pipeline):
        compute_pipeline.request_batch(total_request)

    return source_pipeline
