"""
dacapo.__init__.py
------------------

This module is used to initialize the dacapo module. It imports several core components of the dacapo library including DaCapoArraySource, DaCapoTargetFilter, GammaAugment, ElasticAugment, RejectIfEmpty, CopyMask, GraphSource and Product.
"""

from .dacapo_array_source import DaCapoArraySource
"""
The DaCapoArraySource module which helps to obtain an array of source files involved in the dacapo project. 
"""

from .dacapo_create_target import DaCapoTargetFilter
"""
The DaCapoTargetFilter module which generates custom target file using various filters.
"""

from .gamma_noise import GammaAugment
"""
The GammaAugment module which helps in augmenting the images with gamma correction.
"""

from .elastic_augment_fuse import ElasticAugment
"""
The ElasticAugment module which provides functionalities for elastic deformations on data.
"""

from .reject_if_empty import RejectIfEmpty
"""
The RejectIfEmpty module which helps to check if the data is empty.
"""

from .copy import CopyMask
"""
The CopyMask module which provides a copy operation on mask files.
"""

from .dacapo_points_source import GraphSource
"""
The GraphSource module which works with source points and graphs used in the project.
"""

from .product import Product
"""
The Product module which implements special types of combinations of products.
"""