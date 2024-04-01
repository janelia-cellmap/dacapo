import sys
import traceback


class NoSuchModule:
    """
    This class is used to raise an exception when a module is not found.

    Attributes:
        __name (str): The name of the module that was not found.
        __traceback_str (str): The traceback string of the exception.
        __exception (Exception): The exception raised.
    Methods:
        __getattr__(item): Raises the exception.

    """

    def __init__(self, name):
        """
        Initializes the NoSuchModule object.

        Args:
            name (str): The name of the module that was not found.
        Examples:
            >>> module = NoSuchModule("module")

        """
        self.__name = name
        self.__traceback_str = traceback.format_tb(sys.exc_info()[2])
        errtype, value = sys.exc_info()[:2]
        self.__exception = errtype(value)

    def __getattr__(self, item):
        """
        Raises the exception.

        Args:
            item: The item to get.
        Raises:
            Exception: The exception raised.
        Examples:
            >>> module.function()

        """
        raise self.__exception
