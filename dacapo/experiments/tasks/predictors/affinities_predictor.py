"""
This module contains the AffinitiesPredictor class, a predictor model for affinities prediction in the funkelab dacapo python library.

Classes:
    AffinitiesPredictor: This is a child class from the Predictor class 
    and it serves as a model for predicting affinities in a given dataset.
"""

class AffinitiesPredictor(Predictor):
    """
    A child class of Predictor that handles the prediction of affinities. It is mainly 
    used during the creation of the model and during training as well.
    
    Attributes:
        neighborhood: A list of neighborhood coordinates.
        lsds: Whether to use the local shape descriptor extractor.
        num_voxels: The number of voxels to use in the shape descriptor.
        downsample_lsds: The factor to downsample the shape descriptors.
        grow_boundary_iterations: The number of iterations to grow the boundaries.
        pwdims: The dimensions of the patch-wise model.
        affs_weight_clipmin: The minimum value to clip weights for affinity balances.
        affs_weight_clipmax: The maximum value to clip weights for affinity balances.
        lsd_weight_clipmin: The minimum value to clip weights for LSD affinity balances.
        lsd_weight_clipmax: The maximum value to clip weights for LSD affinity balances.
        background_as_object: Whether to treat the background as an object.
    """
    
    def extractor(self, voxel_size):
        """
        Method to create an LsdExtractor object for the given voxel size.
        Args:
            voxel_size: The size of the voxel.
        """

    def dims(self):
        """
        Method to grab the dimensions of the provided coordinate neighborhood size.
        """
    
    def sigma(self, voxel_size):
        """
        Method to compute the sigma for the Gaussian smoothing using the voxel size.
        Args:
            voxel_size: The size of the voxel.
        """

    def lsd_pad(self, voxel_size):
        """
        Method to compute the padding required for LSD extraction using the voxel size.
        Args:
            voxel_size: The size of the voxel.
        """

    def num_channels(self):
        """
        Method to compute the number of channels. It returns the sum of the number of neighborhood
        entries and LSD descriptors, if LSD is enabled.
        """

    def create_model(self, architecture):
        """
        Method to create a model architecture with the appropriate architecture for predicting affinities.
        Args:
            architecture : The architecture of the model.
        """

    def create_target(self, gt):
        """
        Method to create a target for affinities prediction.
        Args:
            gt: The segmentation ground truth to be used.
        """
    
    def _grow_boundaries(self, mask, slab):
        """
        Method to grow boundaries on a given mask.
        Args:
            mask:
            slab: 
        """

    def create_weight(self, gt, target, mask, moving_class_counts=None):
        """
        This method creates a weight mask for the model.
        Args:
            gt: 
            target:
            mask:
            moving_class_counts (Optional): 
        """

    def gt_region_for_roi(self, target_spec):
        """
        This method defines the region of interest for AffinitiesPredictor
        Args:
            target_spec: Target specification for the region.
        """

    @property
    def output_array_type(self):
        """
        This method sets the output array type for AffinitiesPredictor.
        """