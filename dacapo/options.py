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
    """
    Parse and return the config options from the YAML files.

    Yaml files are parsed in the order of their precedence (highest first).

    Returns:
        dict: Dictionary containing all the parsed options.
    """
    for path in options_files:
        if not path.exists():
            continue

        with path.open("r") as f:
            return yaml.safe_load(f)


class Options:
    """
    Singleton class used to hold and access parsed configuration options.
    """
    _instance = None

    def __init__(self):
        """
        Constructor method is private to enforce Singleton pattern.
        
        Raises:
            RuntimeError: Always raises this error as it's a Singleton.
        """
        raise RuntimeError("Singleton: Use Options.instance()")
    
    @classmethod
    def instance(cls, **kwargs):
        """
        Get the singleton instance of the Options class.
        
        Args:
            **kwargs: Optional named arguments to parse as options.
            
        Returns:
            Options: The singleton instance of Options.
        """
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.__parse_options(**kwargs)

        return cls._instance

    def __getattr__(self, name):
        """
        Get an option by its name.
        
        Args:
            name (str): The name of the option.
            
        Returns:
            Any: The value of the option.
            
        Raises:
            RuntimeError: If the requested option does not exist.
        """
        try:
            return self.__options[name]
        except KeyError:
            raise RuntimeError(
                f"Configuration file {self.filename} does not contain an "
                f"entry for option {name}"
            )

    def __parse_options(self, **kwargs):
        """
        Private method to parse and set the configuration options.
        
        Args:
            **kwargs: Optional named arguments to parse as options.
        """
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
