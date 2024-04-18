from .local_array_store import LocalArrayStore
from .local_weights_store import LocalWeightsStore
from .s3_weights_store import S3WeightsStore
from .mongo_config_store import MongoConfigStore
from .file_config_store import FileConfigStore
from .mongo_stats_store import MongoStatsStore
from .file_stats_store import FileStatsStore
from dacapo import Options

from upath import UPath as Path


def create_config_store():
    """
    Create a config store based on the global DaCapo options.

    Returns:
        ConfigStore: The created config store.
    Raises:
        ValueError: If the store type is not supported.
    Examples:
        >>> create_config_store()
        <dacapo.store.file_config_store.FileConfigStore object at 0x7f2e4f8e9e80>
    Note:
        Currently, only the FileConfigStore and MongoConfigStore are supported.
    """

    options = Options.instance()

    if options.type == "mongo":
        db_host = options.mongo_db_host
        db_name = options.mongo_db_name
        return MongoConfigStore(db_host, db_name)
    elif options.type == "files":
        store_path = Path(options.runs_base_dir).expanduser()
        return FileConfigStore(store_path / "configs")
    else:
        raise ValueError(f"Unknown store type {options.type}")


def create_stats_store():
    """
    Create a statistics store based on the global DaCapo options.

    Args:
        options (Options): The global DaCapo options.
    Returns:
        StatsStore: The created statistics store.
    Raises:
        ValueError: If the store type is not supported.
    Examples:
        >>> create_stats_store()
        <dacapo.store.file_stats_store.FileStatsStore object at 0x7f2e4f8e9e80>
    Note:
        Currently, only the FileStatsStore and MongoStatsStore are supported.

    """

    options = Options.instance()

    if options.type == "mongo":
        db_host = options.mongo_db_host
        db_name = options.mongo_db_name
        return MongoStatsStore(db_host, db_name)
    elif options.type == "files":
        store_path = Path(options.runs_base_dir).expanduser()
        return FileStatsStore(store_path / "stats")
    else:
        raise ValueError(f"Unknown store type {options.type}")


def create_weights_store():
    """
    Create a weights store based on the global DaCapo options.

    Args:
        options (Options): The global DaCapo options.
    Returns:
        WeightsStore: The created weights store.
    Examples:
        >>> create_weights_store()
        <dacapo.store.local_weights_store.LocalWeightsStore object at 0x7f2e4f8e9e80>
    Note:
        Currently, only the LocalWeightsStore is supported.
    """

    options = Options.instance()

    if options.store == "s3":
        s3_bucket = options.s3_bucket
        return S3WeightsStore(s3_bucket)
    elif options.store == "local":
        base_dir = Path(options.runs_base_dir).expanduser()
        return LocalWeightsStore(base_dir)
    else:
        raise ValueError(f"Unknown weights store type {options.type}")


def create_array_store():
    """
    Create an array store based on the global DaCapo options.

    Args:
        options (Options): The global DaCapo options.
    Returns:
        ArrayStore: The created array store.
    Raises:
        ValueError: If the store type is not supported.
    Examples:
        >>> create_array_store()
        <dacapo.store.local_array_store.LocalArrayStore object at 0x7f2e4f8e9e80>
    Note:
        Currently, only the LocalArrayStore is supported.
    """

    options = Options.instance()

    # currently, only the LocalArrayStore is supported
    base_dir = Path(options.runs_base_dir).expanduser()
    return LocalArrayStore(base_dir)
