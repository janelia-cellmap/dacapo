from .augment_config import AugmentConfig

import gunpowder as gp

import attr


@attr.s
class IntensityScaleShiftAugmentConfig(AugmentConfig):
    """
    This class is an implementation of AugmentConfig that applies intensity scaling and shifting.

    Attributes:
        scale (float): A constant to scale your intensities.
        shift (float): A constant to shift your intensities.
    Methods:
        node(raw_key, _gt_key=None, _mask_key=None): Get a gp.IntensityScaleShift node.
    Note:
        This class is a subclass of AugmentConfig.
    """

    scale: float = attr.ib(
        metadata={"help_text": "A constant to scale your intensities."}
    )
    shift: float = attr.ib(
        metadata={"help_text": "A constant to shift your intensities."}
    )

    def node(self, raw_key: gp.ArrayKey, _gt_key=None, _mask_key=None):
        """
        Get a gp.IntensityScaleShift node.

        Args:
            raw_key (gp.ArrayKey): Key for raw data.
            _gt_key ([type], optional): Specific key for ground truth data, not used in this implementation. Defaults to None.
            _mask_key ([type], optional): Specific key for mask data, not used in this implementation. Defaults to None.
        Returns:
            gunpowder.IntensityScaleShift : Intensity scaling and shifting node which can be incorporated in the pipeline.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> node = intensity_scale_shift_augment_config.node(raw_key)
        """
        return gp.IntensityScaleShift(raw_key, scale=self.scale, shift=self.shift)
