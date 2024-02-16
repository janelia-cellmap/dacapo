```python
from .dataset import Dataset
from .arrays import Array

class DummyDataset(Dataset):
    """DummyDataset is a child class of the Dataset. This class has property 'raw' of Array type and a name.

    Args:
        dataset_config (object): an instance of a configuration class.
    """
    
    raw: Array

    def __init__(self, dataset_config):
        """Initializes the array type 'raw' and name for the DummyDataset instance.

        Args:
            dataset_config (object): an instance of a configuration class that includes the name and
            raw configuration of the data.
        """
        super().__init__()
        self.name = dataset_config.name
        self.raw = dataset_config.raw_config.array_type(dataset_config.raw_config)
```