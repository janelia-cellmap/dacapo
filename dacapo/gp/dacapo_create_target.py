# from dacapo.stateless.arraysources.helpers import ArraySource

from dacapo.experiments.tasks.predictors import Predictor

import gunpowder as gp


class DaCapoTargetProvider(gp.BatchFilter):
    """A Gunpowder node for generating the target from the ground truth

    Args:

        Predictor (Predictor):

            The DaCapo Predictor to use to transform gt into target

        gt_key (``gp.ArrayKey``):

            The key to use to fetch ground truth data.

        target_key (``gp.ArrayKey``):

            The key with which to provide the target.
    """

    def __init__(
        self, predictor: Predictor, gt_key: gp.ArrayKey, target_key: gp.ArrayKey
    ):
        self.predictor = predictor
        self.gt_key = gt_key
        self.target_key = target_key

    def setup(self):
        # TODO: How to update the spec?
        raise NotImplementedError()

    def prepare(self, request):
        deps = gp.BatchRequest()
        target_spec = request[self.target_key].copy()
        target_spec.roi = self.predictor.gt_region_for_roi(target_spec.roi)
        deps[self.gt_key] = target_spec
        return deps

    def process(self, batch, request):
        output = gp.Batch()
        gt_data = batch[self.gt_key].data
        target_data = self.predictor.create_target(gt_data)

        raise NotImplementedError("How to define target spec?")
        output[self.target_key] = gp.Array(target_data, target_spec)

        return output
