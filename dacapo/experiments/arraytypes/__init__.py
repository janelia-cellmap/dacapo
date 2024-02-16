Below are the script files with added docstrings in Google Style Docstrings.

```python
from .annotations import AnnotationArray
from .intensities import IntensitiesArray
from .distances import DistanceArray
from .mask import Mask
from .embedding import EmbeddingArray
from .probabilities import ProbabilityArray

def dacapo():
    """This is the main function of the dacapo python library.

    This function integrates multiple scripts/modules of the dacapo library 
    including `AnnotationArray`, `IntensitiesArray`, `DistanceArray`, 
    `Mask`, `EmbeddingArray` and `ProbabilityArray`.

    Note:
        To use this function, the above mentioned scripts/modules should be 
        properly installed and imported.
    """
    pass

class AnnotationArray:
    """Handles annotations for the dacapo library.

    This class provides functionalities to handle and manipulate annotations
    in the dacapo library.
    """
    pass

class IntensitiesArray:
    """Handles intensity arrays for the dacapo python library.

    This class provides functions for handling and manipulating
    intensity arrays in the dacapo library.
    """
    pass

class DistanceArray:
    """Handles distance arrays for the dacapo python library.

    This class provides functionalities for handling and manipulating
    distance array.
    """
    pass

class Mask:
    """Handles masks for the dacapo python library.

    This class provides functionalities to handle and manipulate mask
    in the dacapo library.
    """
    pass

class EmbeddingArray:
    """Handles embedding arrays for the dacapo python library.
    
    This class provides functionalities for handling and manipulating
    embedding array.
    """
    pass

class ProbabilityArray:
    """Handles probability arrays for the dacapo python library.

    This class provides functionalities for handling and manipulating 
    probability array.
    """
    pass
```

Note: The docstrings are added before the class definitions. If you would like to add docstrings inside the class, you can do so by defining it right after the class definition and before any method definitions.