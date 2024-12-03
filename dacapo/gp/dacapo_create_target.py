from dacapo.experiments.tasks.predictors import Predictor
from dacapo.tmp import gp_to_funlib_array

import gunpowder as gp

from typing import Optional


class DaCapoTargetFilter(gp.BatchFilter):
    """
    A Gunpowder node for generating the target from the ground truth

    Attributes:
        Predictor (Predictor):
            The DaCapo Predictor to use to transform gt into target
        gt (``Array``):
            The dataset to use for generating the target.
        target_key (``gp.ArrayKey``):
            The key with which to provide the target.
        weights_key (``gp.ArrayKey``):
            The key with which to provide the weights.
        mask_key (``gp.ArrayKey``):
            The key with which to provide the mask.
    Methods:
        setup(): Set up the provider.
        prepare(request): Prepare the request.
        process(batch, request): Process the batch.
    Note:
        This class is a subclass of gunpowder.BatchFilter and is used to
        generate the target from the ground truth.
    """

    def __init__(
        self,
        predictor: Predictor,
        gt_key: gp.ArrayKey,
        target_key: Optional[gp.ArrayKey] = None,
        weights_key: Optional[gp.ArrayKey] = None,
        mask_key: Optional[gp.ArrayKey] = None,
    ):
        """
        Initialize the DacapoCreateTarget object.

        Args:
            predictor (Predictor): The predictor object used for prediction.
            gt_key (gp.ArrayKey): The ground truth key.
            target_key (Optional[gp.ArrayKey]): The target key. Defaults to None.
            weights_key (Optional[gp.ArrayKey]): The weights key. Defaults to None.
            mask_key (Optional[gp.ArrayKey]): The mask key. Defaults to None.
        Raises:
            AssertionError: If neither target_key nor weights_key is provided.
        Examples:
            >>> from dacapo.experiments.tasks.predictors import Predictor
            >>> from gunpowder import ArrayKey
            >>> from gunpowder import ArrayKey
            >>> from gunpowder import ArrayKey
            >>> predictor = Predictor()
            >>> gt_key = ArrayKey("GT")
            >>> target_key = ArrayKey("TARGET")
            >>> weights_key = ArrayKey("WEIGHTS")
            >>> mask_key = ArrayKey("MASK")
            >>> target_filter = DaCapoTargetFilter(predictor, gt_key, target_key, weights_key, mask_key)
        Note:
            The target filter is used to generate the target from the ground truth.

        """
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
        """
        Set up the provider. This function sets the provider to provide the
        target with the given key.

        Raises:
            RuntimeError: If the key is already provided.
        Examples:
            >>> target_filter.setup()

        """
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
        """
        Prepare the request.

        Args:
            request (gp.BatchRequest): The request to prepare.
        Returns:
            deps (gp.BatchRequest): The dependencies.
        Raises:
            NotImplementedError: If the target_key is not provided.
        Examples:
            >>> request = gp.BatchRequest()
            >>> request[gp.ArrayKey("GT")] = gp.ArraySpec(roi=gp.Roi((0, 0, 0), (1, 1, 1)))
            >>> target_filter.prepare(request)

        """
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
        """
        Process the batch.

        Args:
            batch (gp.Batch): The batch to process.
            request (gp.BatchRequest): The request to process.
        Returns:
            output (gp.Batch): The output batch.
        Examples:
            >>> request = gp.BatchRequest()
            >>> request[gp.ArrayKey("GT")] = gp.ArraySpec(roi=gp.Roi((0, 0, 0), (1, 1, 1)))
            >>> target_filter.process(batch, request)
        """
        output = gp.Batch()

        gt_array = gp_to_funlib_array(batch[self.gt_key])
        target_array = self.predictor.create_target(gt_array)
        mask_array = gp_to_funlib_array(batch[self.mask_key])

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
