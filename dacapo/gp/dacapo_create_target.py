from dacapo.experiments.tasks.predictors import Predictor
from dacapo.experiments.datasplits.datasets.arrays import NumpyArray

import gunpowder as gp

from typing import Optional


class DaCapoTargetFilter(gp.BatchFilter):
    """A Gunpowder node for generating the target from the ground truth

    Args:

        Predictor (Predictor):

            The DaCapo Predictor to use to transform gt into target

        gt (``Array``):

            The dataset to use for generating the target.

        target_key (``gp.ArrayKey``):

            The key with which to provide the target.
    """

    def __init__(
        self,
        predictor: Predictor,
        gt_key: gp.ArrayKey,
        target_key: Optional[gp.ArrayKey] = None,
        weights_key: Optional[gp.ArrayKey] = None,
        mask_key: Optional[gp.ArrayKey] = None,
    ):
        self.predictor = predictor
        self.gt_key = gt_key
        self.target_key = target_key
        self.weights_key = weights_key
        self.mask_key = mask_key

        self.moving_counts = None

        assert (
            target_key is not None or weights_key is not None
        ), "Must provide either target or weights"

    def setup(self):
        provided_spec = gp.ArraySpec(
            roi=self.spec[self.gt_key].roi,
            voxel_size=self.spec[self.gt_key].voxel_size,
            interpolatable=self.predictor.output_array_type.interpolatable,
        )
        if self.target_key is not None:
            self.provides(self.target_key, provided_spec)

        provided_spec = gp.ArraySpec(
            roi=self.spec[self.gt_key].roi,
            voxel_size=self.spec[self.gt_key].voxel_size,
            interpolatable=True,
        )
        if self.weights_key is not None:
            self.provides(self.weights_key, provided_spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        # TODO: Does the gt depend on weights too?
        request_spec = None
        if self.target_key is not None:
            request_spec = request[self.target_key]
            request_spec.voxel_size = self.spec[self.gt_key].voxel_size
            request_spec = self.predictor.gt_region_for_roi(request_spec)
        elif self.weights_key is not None:
            request_spec = request[self.weights_key].copy()
        else:
            raise NotImplementedError("Should not be reached!")
        assert request_spec is not None
        deps[self.gt_key] = request_spec
        if self.mask_key is not None:
            deps[self.mask_key] = request_spec
        return deps

    def process(self, batch, request):
        output = gp.Batch()

        gt_array = NumpyArray.from_gp_array(batch[self.gt_key])
        target_array = self.predictor.create_target(gt_array)
        mask_array = NumpyArray.from_gp_array(
            batch[self.mask_key]
        )  # TODO: doesn't this require mask_key to be set?

        if self.target_key is not None:
            request_spec = request[self.target_key]
            request_spec.voxel_size = gt_array.voxel_size
            output[self.target_key] = gp.Array(
                target_array[request_spec.roi], request_spec
            )
        if self.weights_key is not None:
            weight_array, self.moving_counts = self.predictor.create_weight(
                gt_array,
                target_array,
                mask=mask_array,
                moving_class_counts=self.moving_counts,
            )
            request_spec = request[self.weights_key]
            request_spec.voxel_size = gt_array.voxel_size
            output[self.weights_key] = gp.Array(
                weight_array[request_spec.roi], request_spec
            )
        return output
