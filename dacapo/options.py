import os
import yaml
import logging
from os.path import expanduser
from pathlib import Path

import attr
from cattr import Converter

from typing import Optional


logger = logging.getLogger(__name__)


@attr.s
class DaCapoConfig:
    db_type: str = attr.ib(
        default="files",
        metadata={
            "help_text": "The type of store to use for storing configurations and statistics. "
            "Currently, only 'files' and 'mongo' are supported with files being the default."
        },
    )
    runs_base_dir: Path = attr.ib(
        default=Path(expanduser("~/.dacapo")),
        metadata={
            "help_text": "The path at DaCapo will use for reading and writing any necessary data."
        },
    )
    compute_context_config: dict = attr.ib(
        default={"type": "LocalTorch", "config": {"device": None}},
        metadata={
            "help_text": "The configuration for the compute context to use. "
            "This is a dictionary with the keys being the names of the compute context and the values being the configuration for that context."
        },
    )
    mongo_db_host: Optional[str] = attr.ib(
        default=None,
        metadata={
            "help_text": "The host of the MongoDB instance to use for storing configurations and statistics."
        },
    )
    mongo_db_name: Optional[str] = attr.ib(
        default=None,
        metadata={
            "help_text": "The name of the MongoDB database to use for storing configurations and statistics."
        },
    )

    def serialize(self):
        converter = Converter()
        return converter.unstructure(self)


class Options:
    def __init__(self):
        raise RuntimeError("Singleton: Use Options.instance()")

    @classmethod
    def instance(cls, **kwargs) -> DaCapoConfig:
        config = cls.__parse_options(**kwargs)

        return config

    @classmethod
    def config_file(cls) -> Optional[Path]:
        env_dict = dict(os.environ)
        if "OPTIONS_FILE" in env_dict:
            options_files = [Path(env_dict["OPTIONS_FILE"])]
        else:
            options_files = []

        # options files in order of precedence (highest first)
        options_files += [
            Path("./dacapo.yaml"),
            Path(expanduser("~/.config/dacapo/dacapo.yaml")),
        ]
        for path in options_files:
            if path.exists():
                return path
        return None

    @classmethod
    def __parse_options_from_file(cls):
        if (config_file := cls.config_file()) is not None:
            with config_file.open("r") as f:
                return yaml.safe_load(f)
        else:
            return {}

    @classmethod
    def __parse_options(cls, **kwargs):
        options = cls.__parse_options_from_file()
        options.update(kwargs)

        converter = Converter()

        return converter.structure(options, DaCapoConfig)
