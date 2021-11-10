from dacapo.experiments.tasks.predictors import Predictor
from dacapo.experiments.datasplits.datasets.arrays import Array, NumpyArray

import gunpowder as gp


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
    ):
        self.predictor = predictor
        self.gt_key = gt_key
        self.target_key = target_key

    def setup(self):
        provided_spec = gp.ArraySpec(
            roi=self.spec[self.gt_key].roi,
            interpolatable=self.predictor.output_array_type.interpolatable,
        )
        self.provides(self.target_key, provided_spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.gt_key] = request[self.target_key].copy()
        return deps

    def process(self, batch, request):
        output = gp.Batch()
        request_spec = request[self.target_key]

        gt_array = NumpyArray.from_gp_array(batch[self.gt_key])
        target_array = self.predictor.create_target(gt_array)

        output[self.target_key] = gp.Array(target_array[request_spec.roi], request_spec)
        return output
