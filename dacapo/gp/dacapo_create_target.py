from dacapo.experiments.tasks.predictors import Predictor
from dacapo.experiments.datasplits.datasets.arrays import Array, NumpyArray

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
        target_key: gp.ArrayKey,
        weights_key: gp.ArrayKey,
        mask_key: Optional[gp.ArrayKey] = None,
    ):
        self.predictor = predictor
        self.gt_key = gt_key
        self.target_key = target_key
        self.weights_key = weights_key
        self.mask_key = mask_key

    def setup(self):
        provided_spec = gp.ArraySpec(
            roi=self.spec[self.gt_key].roi,
            voxel_size=self.spec[self.gt_key].voxel_size,
            interpolatable=self.predictor.output_array_type.interpolatable,
        )
        self.provides(self.target_key, provided_spec)

        provided_spec = gp.ArraySpec(
            roi=self.spec[self.gt_key].roi,
            voxel_size=self.spec[self.gt_key].voxel_size,
            interpolatable=True,
        )
        self.provides(self.weights_key, provided_spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        # TODO: Does the gt depend on weights too?
        deps[self.gt_key] = request[self.target_key].copy()
        if self.mask_key is not None:
            deps[self.mask_key] = request[self.target_key].copy()
        return deps

    def process(self, batch, request):
        output = gp.Batch()

        gt_array = NumpyArray.from_gp_array(batch[self.gt_key])
        target_array = self.predictor.create_target(gt_array)
        weight_array = self.predictor.create_weight(gt_array, target_array)

        if self.mask_key is not None:
            mask_array = NumpyArray.from_gp_array(batch[self.mask_key])
            weight_array = NumpyArray.from_np_array(
                weight_array[weight_array.roi] * mask_array[mask_array.roi],
                weight_array.roi,
                weight_array.voxel_size,
                weight_array.axes,
            )

        request_spec = request[self.target_key]
        request_spec.voxel_size = gt_array.voxel_size
        output[self.target_key] = gp.Array(target_array[request_spec.roi], request_spec)
        request_spec = request[self.weights_key]
        request_spec.voxel_size = gt_array.voxel_size
        output[self.weights_key] = gp.Array(
            weight_array[request_spec.roi], request_spec
        )
        return output
