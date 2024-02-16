"""
This script defines a class 'OneHotPredictor' which extends the 'Predictor' class. This class has methods and properties responsible for creating models, targets and weights, determining array type outputs, and processing labels into one hot encoded arrays.

Classes:
    OneHotPredictor: Predictor class extended for handling one hot encoding specifications on the 'classes' input parameter.

"""

class OneHotPredictor(Predictor):
    """
    This class extends the Predictor class and it applies the functions of the Predictor to a list of class labels. It specifically handles the conversion of class labels into one hot-encoded format.
    
    Attributes:
        classes (List[str]): Label data to apply one-hot encoding to.
    """

    def __init__(self, classes: List[str]):
        """
        Initializes the predictor classes.

        Args:
            classes (List[str]): Label data to apply one-hot encoding to.
        """
        
        self.classes = classes

    @property
    def embedding_dims(self):
        """
        Returns the count of classes.
        
        Returns:
            int: The length will give the dimension of the embedding.
        """
        return len(self.classes)

    def create_model(self, architecture):
        """
        Creates the 3D Convolution layer model of the data.

        Args:
            architecture: The architecture setup for the number of output channels.

        Returns:
            Model: Returns the 3D Convolution layer connected to the outputs.
        """
        
        return Model(architecture, head)

    def create_target(self, gt):
        """
        Returns a numpy array object from the one hot-encoded data.

        Args:
            gt: The ground truth object to get the voxel size, roi, and axes.

        Returns:
            NumpyArray: The array class object made after the one hot encoding process.
        """
        
        return NumpyArray.from_np_array(
            one_hots,
            gt.roi,
            gt.voxel_size,
            gt.axes,
        )

    def create_weight(self, gt, target, mask, moving_class_counts=None):
        """
        Returns the numpy array with weights of the target.

        Args:
            gt: The ground truth object.
            target: The object created as the target for the model.
            mask: The masking of the data.
            moving_class_counts (optional): the class counts moving across the data.

        Returns:
            numpy array: Returns a tuple with the array object with the weights and target with 'None'.
        """
        
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
        Returns the probability array of the classes.

        Returns:
            ProbabilityArray: Returns the object of the 'ProbabilityArray' of the classes.
        """
        
        return ProbabilityArray(self.classes)

    def process(
        self,
        labels: np.ndarray,
    ):
        """
        Returns the one-hot encoded array of the label data.

        Args:
            labels (np.ndarray): The array to convert into one-hot encoding.

        Returns:
            np.ndarray: The one-hot encoded numpy array.
        """
        
        return one_hots
