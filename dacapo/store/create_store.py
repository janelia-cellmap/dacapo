Your docstrings have been added. Here is the modified code:

```python
from .local_array_store import LocalArrayStore
from .local_weights_store import LocalWeightsStore
from .mongo_config_store import MongoConfigStore
from .file_config_store import FileConfigStore
from .mongo_stats_store import MongoStatsStore
from .file_stats_store import FileStatsStore
from dacapo import Options

from pathlib import Path


def create_config_store():
    """
    Create and return a configuration store. The type of store is based on the global DaCapo options.
    
    Raises:
        ValueError: If the store type is not recognized.
        
    Returns:
        MongoConfigStore or FileConfigStore: The instantiated configuration store object.
    """

    options = Options.instance()

    try:
        store_type = options.type
    except RuntimeError:
        store_type = "files"
    if store_type == "mongo":
        db_host = options.mongo_db_host
        db_name = options.mongo_db_name
        return MongoConfigStore(db_host, db_name)
    elif store_type == "files":
        store_path = Path(options.runs_base_dir).expanduser()
        return FileConfigStore(store_path / "configs")
    else:
        raise ValueError(f"Unknown store type {store_type}")


def create_stats_store():
    """
    Create and return a statistics store. The type of store is based on the global DaCapo options.
    
    Returns:
        MongoStatsStore or FileStatsStore: The instantiated statistic store object.
    """

    options = Options.instance()

    try:
        store_type = options.type
    except RuntimeError:
        store_type = "mongo"
    if store_type == "mongo":
        db_host = options.mongo_db_host
        db_name = options.mongo_db_name
        return MongoStatsStore(db_host, db_name)
    elif store_type == "files":
        store_path = Path(options.runs_base_dir).expanduser()
        return FileStatsStore(store_path / "stats")


def create_weights_store():
    """
    Create and return a weights store. The type of store is based on the global DaCapo options.
    Currently, only the LocalWeightsStore is supported.
    
    Returns:
        LocalWeightsStore: The instantiated weights store object.
    """
    
    options = Options.instance()

    # currently, only the LocalWeightsStore is supported
    base_dir = Path(options.runs_base_dir).expanduser()
    return LocalWeightsStore(base_dir)


def create_array_store():
    """
    Create and return an array store. The type of store is based on the global DaCapo options.
    Currently, only the LocalArrayStore is supported.
    
    Returns:
        LocalArrayStore: The instantiated array store object.
    """
    
    options = Options.instance()

    # currently, only the LocalArrayStore is supported
    base_dir = Path(options.runs_base_dir).expanduser()
    return LocalArrayStore(base_dir)
```