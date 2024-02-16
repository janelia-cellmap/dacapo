"""
This python file defines a DummyPredictor class which inherits from the Predictor class in dacapo library.

The DummyPredictor class allows the user to create a machine learning model, define target and weight, and set the output
array type for the Predictor. Note that the target and weight creation process utilized here are for demonstration 
purposes and do not reflect any practical setting in real-world scenarios. 

This class takes an integer as parameter which assists in defining various processes in the class.
"""

class DummyPredictor(Predictor):
    """Main class of the module, which utilized to define and manipulate features of predicted data."""

    def __init__(self, embedding_dims):
        """
        Initializes the DummyPredictor. 

        Args:
            embedding_dims: An integer indicating the dimension of the embedding vector.
        """
        self.embedding_dims = embedding_dims

    def create_model(self, architecture):
        """
        Creates a Conv3d model based on the given architecture.

        Args:
            architecture: The architecture of the Convolutional Neural Network.

        Returns:
            A Model object based on the given architecture and a Conv3d.
        """
        # Conv3d
        head = torch.nn.Conv3d(
            architecture.num_out_channels, self.embedding_dims, kernel_size=3
        )

        return Model(architecture, head)

    def create_target(self, gt):
        """
        Function to create a target numpy array of zeros based on the ground truth data dimensions.

        Args:
            gt: The ground truth data.

        Returns:
            A numpy array of zeros, created based on the ground truth data dimensions.
        """
        # zeros
        return NumpyArray.from_np_array(
            np.zeros((self.embedding_dims,) + gt.data.shape[-gt.dims :]),
            gt.roi,
            gt.voxel_size,
            ["c"] + gt.axes,
        )

    def create_weight(self, gt, target, mask, moving_class_counts=None):
        """
        Create weights for the Predictor. The weights are numpy array of ones.

        Args:
            gt: The ground truth data.
            target: The target for the Predictor.
            mask: Mask for the ground truth data.
            moving_class_counts (optional): Number of moving classes.

        Returns:
            A tuple containing a numpy array of ones and None.
        """
        # ones
        return (
            NumpyArray.from_np_array(
                np.ones(target.data.shape),
                target.roi,
                target.voxel_size,
                target.axes,
            ),
            None,
        )

    @property
    def output_array_type(self):
        """
        Set the output array type for the Predictor

        Returns:
            The EmbeddingArray with the desired embedding dimensions.
        """
        return EmbeddingArray(self.embedding_dims)