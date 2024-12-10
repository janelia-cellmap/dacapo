import os
import yaml
import logging
from os.path import expanduser
from upath import UPath as Path

import attr
from cattr import Converter

from typing import Optional

logger = logging.getLogger(__name__)


@attr.s
class DaCapoConfig:
    """
    Configuration class for DaCapo.

    Attributes:
        type (str): The type of store to use for storing configurations and statistics.
        runs_base_dir (Path): The path at DaCapo will use for reading and writing any necessary data.
        compute_context (dict): The configuration for the compute context to use.
        mongo_db_host (Optional[str]): The host of the MongoDB instance to use for storing configurations and statistics.
        mongo_db_name (Optional[str]): The name of the MongoDB database to use for storing configurations and statistics.
    Methods:
        serialize: Serialize the DaCapoConfig object.

    """

    type: str = attr.ib(
        default="files",
        metadata={
            "help_text": "The type of store to use for storing configurations and statistics. "
            "Currently, only 'files' and 'mongo' are supported with files being the default."
        },
    )
    runs_base_dir: Path = attr.ib(
        default=Path(expanduser("~/dacapo")),
        metadata={
            "help_text": "The path at DaCapo will use for reading and writing any necessary data. This should be an absolute path."
        },
    )
    compute_context: dict = attr.ib(
        default={"type": "LocalTorch", "config": {}},
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
        """
        Serialize the DaCapoConfig object.

        Returns:
            dict: The serialized representation of the DaCapoConfig object.
        Examples:
            >>> config = DaCapoConfig()
            >>> config.serialize()
            {'type': 'files', 'runs_base_dir': '/home/user/dacapo', 'compute_context': {'type': 'LocalTorch', 'config': {}}, 'mongo_db_host': None, 'mongo_db_name': None}
        """
        converter = Converter()
        data = converter.unstructure(self)
        return {k: v for k, v in data.items() if v is not None}


class Options:
    """
    A class that provides options for configuring DaCapo.

    This class is designed as a singleton and should be accessed using the `instance` method.

    Methods:
        instance: Returns an instance of the Options class.
        config_file: Returns the path to the configuration file.
        __parse_options_from_file: Parses options from the configuration file.
        __parse_options: Parses options from the configuration file and updates them with the provided kwargs.
    """

    def __init__(self):
        """
        Initializes the Options class.

        Raises:
            RuntimeError: If the constructor is called directly instead of using Options.instance().
        Examples:
            >>> options = Options()
            Traceback (most recent call last):
                ...
            RuntimeError: Singleton: Use Options.instance()
        """
        raise RuntimeError("Singleton: Use Options.instance()")

    @classmethod
    def instance(cls, **kwargs) -> DaCapoConfig:
        """
        Returns an instance of the Options class.

        Args:
            kwargs: Additional keyword arguments to update the options.
        Returns:
            An instance of the DaCapoConfig class.
        Examples:
            >>> options = Options.instance()
            >>> options
            DaCapoConfig(type='files', runs_base_dir=PosixPath('/home/user/dacapo'), compute_context={'type': 'LocalTorch', 'config': {}}, mongo_db_host=None, mongo_db_name=None)
        """
        config = cls.__parse_options(**kwargs)
        return config

    @classmethod
    def config_file(cls) -> Optional[Path]:
        """
        Returns the path to the configuration file.

        Returns:
            The path to the configuration file if found, otherwise None.
        Examples:
            >>> Options.config_file()
            PosixPath('/home/user/.config/dacapo/dacapo.yaml')
        """
        env_dict = dict(os.environ)
        if "DACAPO_OPTIONS_FILE" in env_dict:
            options_files = [Path(env_dict["DACAPO_OPTIONS_FILE"])]
        else:
            options_files = []

        # options files in order of precedence (highest first)
        options_files += [
            Path("./dacapo.yaml"),
            Path("~/dacapo.yaml"),
            Path(expanduser("~/.config/dacapo/dacapo.yaml")),
            Path(Path(__file__).parent.parent, "dacapo.yaml"),
        ]
        for path in options_files:
            if path.exists():
                os.environ["DACAPO_OPTIONS_FILE"] = str(path)
                return path
        return None

    @classmethod
    def __parse_options_from_file(cls):
        """
        Parses options from the configuration file.

        Returns:
            A dictionary containing the parsed options.
        Examples:
            >>> Options.__parse_options_from_file()
            {'type': 'files', 'runs_base_dir': '/home/user/dacapo', 'compute_context': {'type': 'LocalTorch', 'config': {}}, 'mongo_db_host': None, 'mongo_db_name': None}
        """
        if (config_file := cls.config_file()) is not None:
            with config_file.open("r") as f:
                return yaml.safe_load(f)
        else:
            return {}

    @classmethod
    def __parse_options(cls, **kwargs):
        """
        Parses options from the configuration file and updates them with the provided kwargs.

        Args:
            kwargs: Additional keyword arguments to update the options.
        Returns:
            A dictionary containing the parsed and updated options.
        Examples:
            >>> Options.__parse_options()
            {'type': 'files', 'runs_base_dir': '/home/user/dacapo', 'compute_context': {'type': 'LocalTorch', 'config': {}}, 'mongo_db_host': None, 'mongo_db_name': None}
        """
        options = cls.__parse_options_from_file()
        options.update(kwargs)

        converter = Converter()

        return converter.structure(options, DaCapoConfig)
