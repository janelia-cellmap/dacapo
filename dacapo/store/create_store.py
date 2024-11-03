from .local_array_store import LocalArrayStore
from .local_weights_store import LocalWeightsStore
from .mongo_config_store import MongoConfigStore
from .file_config_store import FileConfigStore
from .mongo_stats_store import MongoStatsStore
from .file_stats_store import FileStatsStore
from dacapo import Options

from upath import UPath as Path


def create_config_store():
    

    options = Options.instance()

    if options.type == "mongo":
        db_host = options.mongo_db_host
        db_name = options.mongo_db_name
        return MongoConfigStore(db_host, db_name)
    elif options.type == "files":
        store_path = Path(options.runs_base_dir)
        return FileConfigStore(store_path / "configs")
    else:
        raise ValueError(f"Unknown store type {options.type}")


def create_stats_store():
    

    options = Options.instance()

    if options.type == "mongo":
        db_host = options.mongo_db_host
        db_name = options.mongo_db_name
        return MongoStatsStore(db_host, db_name)
    elif options.type == "files":
        store_path = Path(options.runs_base_dir)
        return FileStatsStore(store_path / "stats")
    else:
        raise ValueError(f"Unknown store type {options.type}")


def create_weights_store():
    

    options = Options.instance()

    base_dir = Path(options.runs_base_dir)
    return LocalWeightsStore(base_dir)


def create_array_store():
    

    options = Options.instance()

    # currently, only the LocalArrayStore is supported
    base_dir = Path(options.runs_base_dir).expanduser()
    return LocalArrayStore(base_dir)
