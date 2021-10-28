from dacapo.experiments.tasks.predictors import Predictor
from dacapo.experiments.datasplits.datasets.arrays import Array

import gunpowder as gp


class DaCapoTargetProvider(gp.BatchProvider):
    """A Gunpowder node for generating the target from the ground truth

    Args:

        Predictor (Predictor):

            The DaCapo Predictor to use to transform gt into target

        gt (``Array``):

            The dataset to use for generating the target.

        target_key (``gp.ArrayKey``):

            The key with which to provide the target.
    """

    def __init__(self, predictor: Predictor, gt: Array, target_key: gp.ArrayKey):
        self.predictor = predictor
        self.gt = gt
        self.target_key = target_key

    def setup(self):
        provided_spec = gp.ArraySpec(
            roi=self.gt.roi,
            interpolatable=self.predictor.output_array_type.interpolatable,
            dtype=self.predictor.output_array_type.dtype,
        )
        self.provides(self.target_key, provided_spec)

    def provide(self, request):
        output = gp.Batch()
        request_spec = request[self.target_key]
        
        gt_data = self.gt[request_spec.roi]
        target_data = self.provider.create_target(gt_data)

        output[self.target_key] = gp.Array(target_data, request_spec)
        return output
