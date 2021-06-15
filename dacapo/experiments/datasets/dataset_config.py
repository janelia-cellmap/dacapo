import attr

from .array_source_config import ArraySourceConfig
from .graph_source_config import GraphSourceConfig
from .keys import DataKey
from typing import Dict, Union


DataSourceConfigs = Dict[
    DataKey,
    Union[ArraySourceConfig, GraphSourceConfig]
]


@attr.s
class DatasetConfig:
    """Configuration class for datasets, to be used to create a ``Dataset``
    instance.
    """

    name: str = attr.ib(
        metadata={
            "help_text":
                "A unique name for this dataset. This will be saved so you "
                "and others can find and reuse this dataset. Keep it short "
                "and avoid special characters."
        }
    )

    train_sources: DataSourceConfigs = attr.ib(
        default=attr.Factory(dict),
        metadata={
            "help_text":
                "A list of data sources to be used for training."
        }
    )

    validate_sources: DataSourceConfigs = attr.ib(
        default=attr.Factory(dict),
        metadata={
            "help_text":
                "A list of data sources to be used for validation."
        }
    )

    predict_sources: DataSourceConfigs = attr.ib(
        default=attr.Factory(dict),
        metadata={
            "help_text":
                "A list of data sources to predict on (optional)."
        }
    )
