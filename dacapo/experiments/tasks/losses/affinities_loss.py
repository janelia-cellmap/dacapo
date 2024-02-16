from mylib import MyClass

class SomeModel:
    def __init__(self, parameter1, parameter2):
        """
        Initialize the instance of SomeModel.

        Args:
            parameter1 (int): The first parameter for SomeModel.
            parameter2 (int): The second parameter for SomeModel.
        """
        self.parameter1 = parameter1
        self.paramater2 = parameter2

    def method1(self, arg1, arg2):
        """
        This is an example of a class method.

        Args:
            arg1 (str): This argument is used for ...
            arg2 (bool): This argument is used to ...

        Returns:
            result (type): Description of the result.
        """
        result = MyClass(arg1, arg2)
        return result

    def method2(self):
        """
        This is another example of a class method.

        Returns:
            bool: Whether the model method2 is successful.
        """
        return True