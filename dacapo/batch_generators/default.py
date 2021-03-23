from funlib.geometry import Coordinate
import gunpowder as gp
import numpy as np
import torch

from dacapo.gp import (
    AddChannelDim,
    RemoveChannelDim,
    TransposeDims,
    Train,
)
from dacapo.padding import compute_padding

import os
import warnings
import logging

logger = logging.getLogger(__name__)


class DefaultBatchGenerator:
    def create_pipeline(self, task, dataset, model, optimizer, outdir, snapshot_every):

        # to compute output shape on the fly. Should probably add helper functions to
        # do this automatically on each supported model.
        backbone = model.instantiate(dataset)

        raw_channels = max(1, dataset.raw.num_channels)
        input_shape = Coordinate(model.input_shape)
        output_shape = Coordinate(backbone.output_shape(input_shape))
        voxel_size = dataset.raw.train.voxel_size

        # switch to world units
        input_size = voxel_size * input_shape
        output_size = voxel_size * output_shape

        # keys for provided datasets
        raw = gp.ArrayKey("RAW")
        gt = gp.ArrayKey("GT")
        if hasattr(dataset, "mask"):
            mask = gp.ArrayKey("MASK")
        else:
            mask = None
        if hasattr(dataset, "nonempty_mask"):
            nonempty_mask = gp.ArrayKey("NONEMPTY_MASK")
        else:
            nonempty_mask = None
        if hasattr(dataset, "specified_locations"):
            specified_locations = gp.GraphKey("SPECIFIED_LOCATIONS")
        else:
            specified_locations = None

        snapshot_dataset_names = {
            raw: "raw",
        }

        predictor_keys = {}
        for predictor in task.predictors:
            name = predictor.name
            predictor_keys[name] = (
                gp.ArrayKey(f"{name.upper()}_TARGET"),
                gp.ArrayKey(f"{name.upper()}_WEIGHT"),
            )

            predictor_target, _ = predictor_keys[name]

            snapshot_dataset_names[predictor_target] = f"{name}_target"

        channel_dims = 0 if raw_channels == 1 else 1

        # compute padding
        # should probably be computed per source.
        # What sort of padding do we really want?
        # How about just better error messages when there's not enough data and
        # user provided padding
        """
        _, _, padding = compute_padding(
            dataset.raw.roi,
            dataset.gt.roi,
            input_size,
            output_size,
            voxel_size,
            padding=dataset.train_padding,
        )
        predictor_padding = Coordinate((0,) * padding.dims)
        for predictor in task.predictors:
            predictor_target, predictor_weights = predictor_keys[name]
            predictor_extra_padding = predictor.add_target(
                gt, predictor_target, predictor_weights, mask
            )[2]
            if predictor_extra_padding is not None:
                predictor_padding = Coordinate(
                    tuple(
                        max(a, b)
                        for a, b in zip(predictor_padding, predictor_extra_padding)
                    )
                )

        extra_in_voxel_fractions = np.asarray(
            predictor_padding, dtype=np.float32
        ) / np.asarray(voxel_size)

        predictor_padding = Coordinate(np.ceil(extra_in_voxel_fractions)) * voxel_size

        # print(f"padding: {padding}")
        if task.padding is not None:
            task_padding = (
                (eval(task.padding) + voxel_size - (1,) * len(voxel_size)) / voxel_size
            ) * voxel_size
            padding += task_padding

        # raise Exception(f"Padding: {padding}, extra: {predictor_padding}")
        """

        # Get source nodes
        raw_sources = dataset.raw.train.get_sources(
            raw, gp.ArraySpec(interpolatable=True)
        )
        gt_sources = dataset.gt.train.get_sources(
            gt, gp.ArraySpec(interpolatable=False)
        )
        # keep a list of all sources, appending the optional ones
        all_sources = [raw_sources, gt_sources]
        # paddings = [padding, padding + predictor_padding]
        source_keys = [raw, gt]
        if mask is not None:
            mask_sources = dataset.mask.train.get_sources(
                mask, gp.ArraySpec(interpolatable=False)
            )
            all_sources.append(mask_sources)
            # paddings.append(predictor_padding)
            source_keys.append(mask)
        if nonempty_mask is not None:
            nonempty_mask_sources = dataset.nonempty_mask.train.get_sources(
                nonempty_mask, gp.ArraySpec(interpolatable=False)
            )
            all_sources.append(nonempty_mask_sources)
            # paddings.append(predictor_padding)
            source_keys.append(nonempty_mask)
        if specified_locations is not None:
            assert nonempty_mask is None, (
                "Can only support either nonempty mask or specified "
                "locations for chosing random locations"
            )
            specified_location_sources = dataset.specified_locations.train.get_sources(
                specified_locations, gp.GraphSpec()
            )
            all_sources.append(specified_location_sources)
            # paddings.append(Coordinate((0,) * padding.dims))

        # Agglomerate source nodes into a pipeline with merge providers and random location

        all_sources = tuple(all_sources)
        pipelines = []
        for i, sources in enumerate(zip(*all_sources)):
            pipeline = sources + gp.MergeProvider()
            # for key, padding in zip(source_keys, paddings):
            #     pipeline += gp.Pad(key, padding)
            if nonempty_mask is not None:
                pipeline += gp.RandomLocation(mask=nonempty_mask, min_masked=0.1)
            elif specified_locations is not None:
                locations_source = specified_location_sources[i]
                with gp.build(locations_source):
                    locations_request = gp.BatchRequest()
                    locations_request[specified_locations] = locations_source.spec[
                        specified_locations
                    ]
                    batch = locations_source.request_batch(locations_request)
                    locations = [
                        node.location for node in batch[specified_locations].nodes
                    ]
                logger.warning(f"Got {len(locations)} specified locations")
                pipeline += gp.SpecifiedLocation(
                    locations=locations, choose_randomly=True
                )
            else:
                pipeline += gp.RandomLocation()
            pipelines.append(pipeline)
        pipeline = tuple(pipelines) + gp.RandomProvider()

        pipeline += gp.Normalize(raw)
        # raw: ([c,] d, h, w)
        # gt: ([c,] d, h, w)

        warnings.warn("Augmentations not yet handled!")
        # for augmentation in eval(task.augmentations):
        #     pipeline += augmentation

        # (don't care about gt anymore)
        # raw: ([c,] d, h, w)
        # target: ([c,] d, h, w)

        # Add predictor nodes to pipeline
        for predictor in task.predictors:
            name = predictor.name
            predictor_target, predictor_weights = predictor_keys[name]
            predictor_target_node, predictor_weights_node, _ = predictor.add_target(
                gt, predictor_target, predictor_weights, mask
            )
            pipeline += predictor_target_node
            if predictor_weights_node is not None:
                if predictor_weights_node is not True:
                    pipeline += predictor_weights_node
                else:
                    # weights are provided, but not by a new node.
                    # should maybe just return a no-op node
                    pass
                snapshot_dataset_names[predictor_weights] = f"{name}_weights"
            else:
                predictor_keys[name] = (predictor_target, None)

        # if there is no channel dimension, add one
        if channel_dims == 0:
            pipeline += AddChannelDim(raw)

        pipeline += gp.PreCache(num_workers=5)

        # stack to create a batch dimension
        pipeline += gp.Stack(optimizer.batch_size)

        warnings.warn("PrintProfiling not handled!")

        # generate request for all necessary inputs to training
        request = gp.BatchRequest()
        request.add(raw, input_size)
        request.add(gt, output_size)
        if nonempty_mask is not None:
            request.add(nonempty_mask, output_size)
        for predictor in task.predictors:
            name = predictor.name
            predictor_target, predictor_weight = predictor_keys[name]
            request.add(predictor_target, output_size)
            if predictor_weight is not None:
                request.add(predictor_weight, output_size)

        return pipeline, request, predictor_keys
