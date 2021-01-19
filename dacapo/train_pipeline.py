from .gp import (
    Squash,
    AddChannelDim,
    RemoveChannelDim,
    TransposeDims,
    Train,
    BinarizeNot,
)
from .padding import compute_padding
import gunpowder as gp
import math
import os


def create_pipeline_3d(
    task, data, predictor, optimizer, batch_size, outdir, snapshot_every
):

    raw_channels = max(1, data.raw.num_channels)
    input_shape = gp.Coordinate(task.model.input_shape)
    output_shape = gp.Coordinate(task.model.output_shape)
    voxel_size = data.raw.train.voxel_size

    # switch to world units
    input_size = voxel_size * input_shape
    output_size = voxel_size * output_shape

    # keys for provided datasets
    raw = gp.ArrayKey("RAW")
    gt = gp.ArrayKey("GT")
    if hasattr(data, "mask"):
        mask = gp.ArrayKey("MASK")
    else:
        mask = None

    # keys for generated datasets
    target = gp.ArrayKey("TARGET")
    weights = gp.ArrayKey("WEIGHTS")

    # keys for predictions
    model_outputs = gp.ArrayKey("MODEL_OUTPUTS")
    model_output_grads = gp.ArrayKey("MODEL_OUT_GRAD")
    prediction = gp.ArrayKey("PREDICTION")
    pred_gradients = gp.ArrayKey("PRED_GRADIENTS")

    snapshot_dataset_names = {
        raw: "raw",
        model_outputs: "model_outputs",
        model_output_grads: "model_out_grad",
        target: "target",
        prediction: "prediction",
        pred_gradients: "pred_gradients",
    }

    aux_keys = {}
    aux_grad_keys = {}
    for name, _, _ in task.aux_tasks:
        aux_keys[name] = (
            gp.ArrayKey(f"{name.upper()}_PREDICTION"),
            gp.ArrayKey(f"{name.upper()}_TARGET"),
            gp.ArrayKey(f"{name.upper()}_WEIGHT"),
        )
        aux_grad_keys[name] = gp.ArrayKey(f"{name.upper()}_PRED_GRAD")

        aux_pred, aux_target, _ = aux_keys[name]

        snapshot_dataset_names[aux_pred] = f"{name}_pred"
        snapshot_dataset_names[aux_target] = f"{name}_target"

        aux_grad = aux_grad_keys[name]
        snapshot_dataset_names[aux_grad] = f"{name}_aux_grad"

    channel_dims = 0 if raw_channels == 1 else 1

    num_samples = data.raw.train.num_samples
    assert num_samples == 0, "Multiple samples for 3D training not yet implemented"

    # compute padding
    _, _, padding = compute_padding(
        data.raw.roi,
        data.gt.roi,
        input_size,
        output_size,
        voxel_size,
        padding=data.train_padding,
    )
    target_node, weights_node, extra_gt_padding = predictor.add_target(
        gt, target, weights, mask
    )
    if extra_gt_padding is None:
        extra_gt_padding = gp.Coordinate((0,) * len(padding))
    for name, aux_predictor, _ in task.aux_tasks:
        _, aux_target, aux_weights = aux_keys[name]
        aux_extra_gt_padding = aux_predictor.add_target(gt, aux_target, aux_weights, mask)[
            2
        ]
        if aux_extra_gt_padding is not None:
            extra_gt_padding = gp.Coordinate(
                tuple(max(a, b) for a, b in zip(extra_gt_padding, aux_extra_gt_padding))
            )
    # print(f"padding: {padding}")
    if task.padding is not None:
        padding += eval(task.padding)

    # raise Exception(f"Padding: {padding}, extra: {extra_gt_padding}")

    raw_sources = data.raw.train.get_sources(raw, gp.ArraySpec(interpolatable=True))
    gt_sources = data.gt.train.get_sources(gt, gp.ArraySpec(interpolatable=False))
    if mask is not None:
        mask_sources = data.mask.train.get_sources(
            mask, gp.ArraySpec(interpolatable=False)
        )
    if isinstance(raw_sources, list):
        assert isinstance(gt_sources, list)
        assert len(raw_sources) == len(gt_sources)
        if mask is not None:
            assert isinstance(mask_sources, list)
            assert len(raw_sources) == len(mask_sources)

            pipeline = (
                tuple(
                    (raw_source, gt_source, mask_source)
                    + gp.MergeProvider()
                    + gp.Pad(raw, padding)
                    + gp.Pad(gt, padding + extra_gt_padding)
                    + gp.Pad(mask, padding + extra_gt_padding)
                    + gp.RandomLocation()
                    for raw_source, gt_source, mask_source in zip(
                        raw_sources, gt_sources, mask_sources
                    )
                )
                + gp.RandomProvider()
            )

        else:
            pipeline = (
                tuple(
                    (raw_source, gt_source)
                    + gp.MergeProvider()
                    + gp.Pad(raw, padding)
                    + gp.Pad(gt, padding + extra_gt_padding)
                    + gp.RandomLocation()
                    for raw_source, gt_source in zip(raw_sources, gt_sources)
                )
                + gp.RandomProvider()
            )

    else:
        assert not isinstance(gt_sources, list)
        if mask is not None:
            assert not isinstance(mask_sources, list)
            pipeline = (
                (raw_sources, gt_sources, mask_sources)
                + gp.MergeProvider()
                + gp.Pad(raw, padding)
                + gp.Pad(gt, padding + extra_gt_padding)
                + gp.Pad(mask, padding + extra_gt_padding)
                + gp.RandomLocation()
            )
        else:
            pipeline = (
                (raw_sources, gt_sources)
                + gp.MergeProvider()
                + gp.Pad(raw, padding)
                + gp.Pad(gt, padding + extra_gt_padding)
                + gp.RandomLocation()
            )

    pipeline += gp.Normalize(raw)
    # raw: ([c,] d, h, w)
    # gt: ([c,] d, h, w)
    for augmentation in eval(task.augmentations):
        pipeline += augmentation
    pipeline += target_node
    # (don't care about gt anymore)
    # raw: ([c,] d, h, w)
    # target: ([c,] d, h, w)
    loss_inputs = []
    if weights_node:
        pipeline += weights_node
        loss_inputs.append({0: prediction, 1: target, 2: weights})
        snapshot_dataset_names[weights] = "weights"
    else:
        loss_inputs.append({0: prediction, 1: target})

    head_outputs = []
    head_gradients = []
    for name, aux_predictor, aux_loss in task.aux_tasks:
        aux_prediction, aux_target, aux_weights = aux_keys[name]
        aux_target_node, aux_weights_node, _ = aux_predictor.add_target(
            gt, aux_target, aux_weights, mask
        )
        pipeline += aux_target_node
        if aux_weights_node is not None:
            aux_keys[name] = (
                aux_prediction,
                aux_target,
                aux_weights,
            )
            if aux_weights_node is not True:
                pipeline += aux_weights_node
            loss_inputs.append({0: aux_prediction, 1: aux_target, 2: aux_weights})
            snapshot_dataset_names[aux_weights] = f"{name}_weights"
        else:
            loss_inputs.append({0: aux_prediction, 1: aux_target})
        head_outputs.append({0: aux_prediction})
        aux_pred_gradient = aux_grad_keys[name]
        head_gradients.append({0: aux_pred_gradient})
    # raw: ([c,] d, h, w)
    # target: ([c,] d, h, w)
    # [weights: ([c,] d, h, w)]
    if channel_dims == 0:
        pipeline += AddChannelDim(raw)
    # raw: (c, d, h, w)
    # target: ([c,] d, h, w)
    # [weights: ([c,] d, h, w)]
    pipeline += gp.PreCache(num_workers=5)
    pipeline += gp.Stack(batch_size)
    # raw: (b, c, d, h, w)
    # target: (b, [c,] d, h, w)
    # [weights: (b, [c,] d, h, w)]
    pipeline += Train(
        model=task.model,
        heads=[("opt", predictor)]
        + [(name, aux_pred) for name, aux_pred, _ in task.aux_tasks],
        losses=[task.loss] + [loss for _, _, loss in task.aux_tasks],
        optimizer=optimizer,
        inputs={"x": raw},
        outputs={0: model_outputs},
        head_outputs=[{0: prediction}] + head_outputs,
        loss_inputs=loss_inputs,
        gradients=[{0: model_output_grads}, {0: pred_gradients}] + head_gradients,
        save_every=1e6,
    )
    # raw: (b, c, d, h, w)
    # target: (b, [c,] d, h, w)
    # [weights: (b, [c,] d, h, w)]
    # prediction: (b, [c,] d, h, w)
    if snapshot_every > 0:
        # get channels first
        pipeline += TransposeDims(raw, (1, 0, 2, 3, 4))
        if predictor.target_channels > 0:
            pipeline += TransposeDims(target, (1, 0, 2, 3, 4))
            if weights_node:
                pipeline += TransposeDims(weights, (1, 0, 2, 3, 4))
        if predictor.prediction_channels > 0:
            pipeline += TransposeDims(prediction, (1, 0, 2, 3, 4))
        # raw: (c, b, d, h, w)
        # target: ([c,] b, d, h, w)
        # [weights: ([c,] b, d, h, w)]
        # prediction: ([c,] b, d, h, w)
        if channel_dims == 0:
            pipeline += RemoveChannelDim(raw)
        # raw: ([c,] b, d, h, w)
        # target: (c, b, d, h, w)
        # [weights: ([c,] b, d, h, w)]
        # prediction: (c, b, d, h, w)
        pipeline += gp.Snapshot(
            dataset_names=snapshot_dataset_names,
            every=snapshot_every,
            output_dir=os.path.join(outdir, "snapshots"),
            output_filename="{iteration}.zarr",
        )
    pipeline += gp.PrintProfilingStats(every=10)

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(gt, output_size)
    request.add(target, output_size)
    for name, _, _ in task.aux_tasks:
        aux_pred, aux_target, aux_weight = aux_keys[name]
        request.add(aux_pred, output_size)
        request.add(aux_target, output_size)
        if aux_weight is not None:
            request.add(aux_weight, output_size)
        aux_pred_grad = aux_grad_keys[name]
        request.add(aux_pred_grad, output_size)
    if weights_node:
        request.add(weights, output_size)
    request.add(prediction, output_size)
    request.add(model_outputs, output_size)
    request.add(model_output_grads, output_size)
    if weights_node:
        request.add(weights, output_size)
    request.add(pred_gradients, output_size)

    return pipeline, request


def create_train_pipeline(
    task, data, predictor, optimizer, batch_size, outdir=".", snapshot_every=1000
):

    task_dims = data.raw.spatial_dims

    if task_dims == 2:
        return create_pipeline_2d(
            task, data, predictor, optimizer, batch_size, outdir, snapshot_every
        )
    elif task_dims == 3:
        return create_pipeline_3d(
            task, data, predictor, optimizer, batch_size, outdir, snapshot_every
        )
    else:
        raise RuntimeError("Training other than 2D/3D not yet implemented")
