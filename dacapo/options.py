import yaml
import logging
from os.path import expanduser
from pathlib import Path

logger = logging.getLogger(__name__)

# options files in order of precedence (highest first)
options_files = [
    Path("./dacapo.yaml"),
    Path(expanduser("~/.config/dacapo/dacapo.yaml")),
]

def parse_options():
    for path in options_files:

        if not path.exists():
            continue

        with path.open("r") as f:
            return yaml.safe_load(f)


class Options:

    _instance = None

    def __init__(self):
        raise RuntimeError("Singleton: Use Options.instance()")

    @classmethod
    def instance(cls, **kwargs):

        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.__parse_options(**kwargs)

        return cls._instance

    def __getattr__(self, name):

        try:
            return self.__options[name]
        except KeyError:
            raise RuntimeError(
                f"Configuration file {self.filename} does not contain an "
                f"entry for option {name}"
            )

    def __parse_options(self, **kwargs):
        if len(kwargs) > 0:
            self.__options = kwargs
            self.filename = "kwargs"
            return

        for path in options_files:

            if not path.exists():
                continue

            with path.open("r") as f:
                self.__options = yaml.safe_load(f)
                self.filename = path

            return

        logger.error(
            "No options file found. Please create any of the following " "files:"
        )
        for path in options_files:
            logger.error("\t%s", path.absolute())

        raise RuntimeError("Could not find a DaCapo options file.")
