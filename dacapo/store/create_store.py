from .local_array_store import LocalArrayStore
from .local_weights_store import LocalWeightsStore
from .mongo_config_store import MongoConfigStore
from .mongo_stats_store import MongoStatsStore
from dacapo import Options


def create_config_store():
    """Create a config store based on the global DaCapo options."""

    options = Options.instance()

    # currently, only the MongoConfigStore is supported
    db_host = options.mongo_db_host
    db_name = options.mongo_db_name
    return MongoConfigStore(db_host, db_name)


def create_stats_store():
    """Create a statistics store based on the global DaCapo options."""

    options = Options.instance()

    # currently, only the MongoConfigStore is supported
    db_host = options.mongo_db_host
    db_name = options.mongo_db_name
    return MongoStatsStore(db_host, db_name)


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
