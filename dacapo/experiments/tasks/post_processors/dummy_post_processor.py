"""
This script provides the implementation of dummy post-processing within the dacapo python library.
It contains the DummyPostProcessor class which inherits from the PostProcessor class.
This class returns an iterable of all possible parameters for post-processing implementation and 
stores some dummy data in the output array.

Classes:
    DummyPostProcessor : A class used for enumerating post processing parameters and storing 
    data.

Methods:
    __init__(self, detection_threshold: float) : initializes the detection_threshold.

    enumerate_parameters(self) -> Iterable[DummyPostProcessorParameters] : returns an iterable 
    containing DummyPostProcessorParameters objects.

    set_prediction(self, prediction_array) : contains pass statement (no operation)

    process(self, parameters, output_array_identifier): stores some dummy data in output_array.
"""

class DummyPostProcessor(PostProcessor):
    """This class inherits the PostProcessor class. It is used for enumerating 
    post processing parameters and storing dummy data in the output array.
    
    Args:
        detection_threshold (float): An initial detection threshold.
      
    """
    def __init__(self, detection_threshold: float):
        self.detection_threshold = detection_threshold

    def enumerate_parameters(self) -> Iterable[DummyPostProcessorParameters]:
        """Enumerate all possible parameters of this post-processor. 
        
        Returns:
            Iterable: Returns an iterable of DummyPostProcessorParameters' instances.

        """

        for i, min_size in enumerate(range(1, 11)):
            yield DummyPostProcessorParameters(id=i, min_size=min_size)

    def set_prediction(self, prediction_array):
        """An empty method that is here to satisfy the interface requirements.
        
        Args:
            prediction_array: The prediction array
        """
        pass

    def process(self, parameters, output_array_identifier):
        """Stores dummy data in the output array.

        Args:
            parameters: The parameters for processing
            output_array_identifier: The identifier for the output array    

        """
        
        # store some dummy data
        f = zarr.open(str(output_array_identifier.container), "a")
        f[output_array_identifier.dataset] = np.ones((10, 10, 10)) * parameters.min_size
