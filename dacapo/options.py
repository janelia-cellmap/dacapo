import yaml
import logging
from os.path import expanduser
from pathlib import Path

logger = logging.getLogger(__name__)

# options files in order of precedence (highest first)
options_files = [
    Path('./dacapo.yaml'),
    Path(expanduser('~/.config/dacapo'))
]


class Options:

    _instance = None

    def __init__(self):
        raise RuntimeError("Singleton: Use Options.instance()")

    @classmethod
    def instance(cls):

        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.__parse_options()

        return cls._instance

    def __getattr__(self, name):

        return self.__options[name]

    def __parse_options(self):

        for path in options_files:

            if not path.exists():
                continue

            with path.open('r') as f:
                self.__options = yaml.safe_load(f)

            return

        logger.error(
            "No options file found. Please create any of the following "
            "files:")
        for path in options_files:
            logger.error("\t%s", path)

        raise RuntimeError("Could not find a DaCapo options file.")
