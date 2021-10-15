from .local_array_store import LocalArrayStore
from .local_weights_store import LocalWeightsStore
from .mongo_config_store import MongoConfigStore
from .mongo_stats_store import MongoStatsStore
from dacapo import Options


def create_config_store():
    """Create a config store based on the global DaCapo options."""

    options = Options.instance()

    try:
        store_type = options.type
    except KeyError:
        store_type = "mongo"
    if store_type == "mongo":
        db_host = options.mongo_db_host
        db_name = options.mongo_db_name
        return MongoConfigStore(db_host, db_name)
    elif store_type == "files":
        store_path = options.runs_base_dir
        return FileConfigStore(store_path)


def create_stats_store():
    """Create a statistics store based on the global DaCapo options."""

    options = Options.instance()

    try:
        store_type = options.type
    except KeyError:
        store_type = "mongo"
    if store_type == "mongo":
        db_host = options.mongo_db_host
        db_name = options.mongo_db_name
        return MongoStatsStore(db_host, db_name)
    elif store_type == "files":
        store_path = options.runs_base_dir
        return FileStatsStore(store_path)


def create_weights_store():
    """Create a weights store based on the global DaCapo options."""

    options = Options.instance()

    # currently, only the LocalWeightsStore is supported
    base_dir = options.runs_base_dir
    return LocalWeightsStore(base_dir)


def create_array_store():
    """Create an array store based on the global DaCapo options."""

    options = Options.instance()

    # currently, only the LocalArrayStore is supported
    base_dir = options.runs_base_dir
    return LocalArrayStore(base_dir)
