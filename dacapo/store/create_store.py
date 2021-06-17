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
