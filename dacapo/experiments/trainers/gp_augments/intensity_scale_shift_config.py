"""
A Python file for the IntensityScaleShiftAugmentConfig class, which is used for scaling and shifting 
the pixel intensity of the raw data. The configuration for the scale and shift is given in the form of 
metadata. The `node` method is used to apply the scale and shift on the raw input data.

Attributes:
    AugmentConfig: A base class that provides the configuration for augmentation.
    scale: Float value for scaling the pixel intensities of the raw data.
    shift: Float value for shifting the pixel intensities of the raw data.

Methods:
    node(raw_key, _gt_key=None, _mask_key=None): A method that takes raw data and applies the intensity scale 
    and shift operation. The method returns the transformed data.
"""

@attr.s
class IntensityScaleShiftAugmentConfig(AugmentConfig):
    scale: float = attr.ib(
        metadata={"help_text": "A constant to scale your intensities."}
    )
    shift: float = attr.ib(
        metadata={"help_text": "A constant to shift your intensities."}
    )

    def node(self, raw_key: gp.ArrayKey, _gt_key=None, _mask_key=None):
        """
        A method that applies the scale and shift operation on the raw data;
        by using the provided scale and shift factor.

        Args:
            raw_key (ArrayKey): The raw data in the form of an array.
            _gt_key (ArrayKey, optional): Ignored for this operation, provided for consistency with other augment functions.
            _mask_key (ArrayKey, optional): Ignored for this operation, provided for consistency with other augment functions.

        Returns:
            gnumpy.ndarry: Transformed data after applying the intensity scaling and shift operation.
        """
        return gp.IntensityScaleShift(raw_key, scale=self.scale, shift=self.shift)