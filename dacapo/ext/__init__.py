import sys
import traceback

class NoSuchModule:
    """
    A custom exception class for handling
    situations when a module specified name does not exist.

    Attributes:
        __name: str, name of the module which does not exist.
        __traceback_str: list, the formatted stack trace at the time of the
            exception. It is captured by the sys and traceback module.
        __exception: Exception, stores exception type along with values.
    """

    def __init__(self, name):
        """
        Args:
            name (str): The name of the not existing module.
        """
        self.__name = name
        self.__traceback_str = traceback.format_tb(sys.exc_info()[2])
        errtype, value = sys.exc_info()[:2]
        self.__exception = errtype(value)

    def __getattr__(self, item):
        """
        Raises an exception when trying to access attributes of the not existing module.

        Args:
            item (str): Name of the attribute.

        Raises:
            __exception: custom exception with the details of the original error.
        """
        raise self.__exception