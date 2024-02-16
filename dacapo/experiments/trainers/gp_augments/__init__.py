```python
"""
funkelab dacapo python library script file.

This script file imports various augment configuration classes from different modules 
into the current namespace.

Classes:
    AugmentConfig:  Basic class for augment configuration with its base properties.
    ElasticAugmentConfig : Config file for elastic augmentations in image processing.
    SimpleAugmentConfig: Basic configuration for simple image augmentations.
    GammaAugmentConfig: Config file for gamma corrections in image augmentations.
    IntensityAugmentConfig: Configurations for intensity based augmentations.
    IntensityScaleShiftAugmentConfig: Configuration for scaling and shifting of image 
    intensity during augmentations.
"""

from .augment_config import AugmentConfig
from .elastic_config import ElasticAugmentConfig
from .simple_config import SimpleAugmentConfig
from .gamma_config import GammaAugmentConfig
from .intensity_config import IntensityAugmentConfig
from .intensity_scale_shift_config import IntensityScaleShiftAugmentConfig
```