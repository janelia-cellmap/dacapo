"""
This script file contains a class ArgmaxPostProcessor which is a subclass
of PostProcessor class. Its purpose is to process a set of parameters and 
predictions and utilize them to run blockwise prediction on a given array 
of data from the daCapo library.

Classes:
--------
ArgmaxPostProcessor -> Subclass of PostProcessor class for applying prediction operations.
"""

class ArgmaxPostProcessor(PostProcessor):
    def __init__(self):
        """
        Initialize the ArgmaxPostProcessor object. This class doesn't take 
        any arguments for initialization.
        """

    def enumerate_parameters(self):
        """
        Enumerate all possible parameters of the post-processor and yield 
        ArgmaxPostProcessorParameters objects with id=1.
        
        Yields:
        -------
        ArgmaxPostProcessorParameters: An instance of PostProcessorParameters.
        """

    def set_prediction(self, prediction_array_identifier):
        """
        Set the prediction array using the provided array identifier.
        
        Parameters:
        -----------
        prediction_array_identifier: Identifier for the array to be predicted.
        """

    def process(
        self,
        parameters,
        output_array_identifier,
        compute_context: ComputeContext | str = LocalTorch(),
        num_workers: int = 16,
        chunk_size: Coordinate = Coordinate((64, 64, 64)),
    ):
        """
        Process the predictions on array data using given parameters and identifiers,
        run blockwise prediction and create an output array.

        Parameters:
        -----------
        parameters: Parameters for the post-processor.
        output_array_identifier: Identifier for array in which the output will be stored.
        compute_context : ComputeContext object or str, optional
            Default is LocalTorch() object.
        num_workers : int, optional
            Number of workers, default is 16.
        chunk_size: Coordinate of the chunk size to be used. Dimension size (64, 64, 64) by default.

        Returns:
        --------  
        output_array: New array with the processed output.
        """
