"""
This script defines the class `IntensityAugmentConfig`, a child of the `AugmentConfig` class. This class represents the
configuration for intensity augmentation which could be used to randomly adjust the intensity scale and add shifts to 
the images in the dataset.

Every instance of this class should have three attributes: `scale`, `shift` and `clip`. `scale` and `shift` are tuples 
of two floats representing the range within which to choose a random scale and shift respectively. `clip` is a Boolean 
that controls whether to clip the modified values to [0, 1] or not.

The need for intensity augmentation arises due to differences in the intensity distributions in the image data resulting
from variations in imaging conditions (e.g., different lighting conditions, different imaging equipment, etc.). 
Performing intensity augmentation during the training of machine learning models can make them invariant to these 
changes in the input data, thus improving their generalization ability. 

Attributes:
    scale (Tuple[float, float]): A range within which to choose a random scale factor.
    shift (Tuple[float, float]): A range within which to choose a random additive shift.
    clip (bool): Set to False if modified values should not be clipped to [0, 1].

Methods:
    node(raw_key: gp.ArrayKey, _gt_key=None, _mask_key=None): Returns the gunpowder node for this augmentation.
"""

@attr.s
class IntensityAugmentConfig(AugmentConfig):
    scale: Tuple[float, float] = attr.ib(
        metadata={"help_text": "A range within which to choose a random scale factor."}
    )
    shift: Tuple[float, float] = attr.ib(
        metadata={
            "help_text": "A range within which to choose a random additive shift."
        }
    )
    clip: bool = attr.ib(
        default=True,
        metadata={
            "help_text": "Set to False if modified values should not be clipped to [0, 1]"
        },
    )

    def node(self, raw_key: gp.ArrayKey, _gt_key=None, _mask_key=None):
        """
        Returns an instance of IntensityAugment configured according to this object's attributes.

        Args:
            raw_key (gp.ArrayKey): The ArrayKey of the raw data to apply the intensity augmentation to.
        
        Returns:
            gp.IntensityAugment: An intensity augmentation gunpowder node, configured according to the attributes of this object.
        """
        return gp.IntensityAugment(
            raw_key,
            scale_min=self.scale[0],
            scale_max=self.scale[1],
            shift_min=self.shift[0],
            shift_max=self.shift[1],
            clip=self.clip,
        )