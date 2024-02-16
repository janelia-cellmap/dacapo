    """
    Define DistanceArray class which inherits from ArrayType.

    This class contains methods and attributes related to the array containing signed distances
    to the nearest boundary voxel for a particular label class. It allows positive distances outside 
    an object and negative inside an object. It also includes a property method for interpolation of the array.
    
    Attributes:
        classes (Dict[int, str]): A dictionary mapping from channel to class on which distances were calculated.
    """

    classes: Dict[int, str] = attr.ib(
        metadata={
            "help_text": "A mapping from channel to class on which distances were calculated"
        }
    )

    @property
    def interpolatable(self) -> bool:
        """
        Assesses if the array is interpolatable.

        Returns:
            bool: True if it's interpolatable, False otherwise.
        """
        return True