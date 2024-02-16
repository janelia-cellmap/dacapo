from .local_array_store import LocalArrayStore
from .local_weights_store import LocalWeightsStore
from .mongo_config_store import MongoConfigStore
from .file_config_store import FileConfigStore
from .mongo_stats_store import MongoStatsStore
from .file_stats_store import FileStatsStore
from dacapo import Options

from pathlib import Path


def create_config_store():
    """Create a config store based on the global DaCapo options."""

    options = Options.instance()

    if options.db_type == "mongo":
        db_host = options.mongo_db_host
        db_name = options.mongo_db_name
        return MongoConfigStore(db_host, db_name)
    elif options.db_type == "files":
        store_path = Path(options.runs_base_dir).expanduser()
        return FileConfigStore(store_path / "configs")
    else:
        raise ValueError(f"Unknown store type {options.db_type}")


def create_stats_store():
    """Create a statistics store based on the global DaCapo options."""

    options = Options.instance()

    if options.db_type == "mongo":
        db_host = options.mongo_db_host
        db_name = options.mongo_db_name
        return MongoStatsStore(db_host, db_name)
    elif options.db_type == "files":
        store_path = Path(options.runs_base_dir).expanduser()
        return FileStatsStore(store_path / "stats")
    else:
        raise ValueError(f"Unknown store type {options.db_type}")


def create_weights_store():
    """Create a weights store based on the global DaCapo options."""

    options = Options.instance()

    # currently, only the LocalWeightsStore is supported
    base_dir = Path(options.runs_base_dir).expanduser()
    return LocalWeightsStore(base_dir)


def create_array_store():
    """Create an array store based on the global DaCapo options."""

    options = Options.instance()

    # currently, only the LocalArrayStore is supported
    base_dir = Path(options.runs_base_dir).expanduser()
    return LocalArrayStore(base_dir)
