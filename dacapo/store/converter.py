from cattr import Converter
from cattr.gen import make_dict_unstructure_fn, make_dict_structure_fn
from .conversion_hooks import register_hooks


class TypedConverter(Converter):
    """A converter that stores and retrieves type information for selected
    class hierarchies. Used to reconstruct a concrete class from unstructured
    data.

    Attributes:
        hooks (Dict[Type, List[Hook]]): A dictionary mapping classes to lists of
            hooks that should be applied to them.
    Methods:
        register_hierarchy: Register a class hierarchy for typed
            structure/unstructure conversion.
        __typed_unstructure: Unstructure an object, adding a '__type__' field
            with the class name.
        __typed_structure: Structure an object, using the '__type__' field to
            determine the class.
    Note:
        This class is a subclass of cattr.Converter, and extends it with the
        ability to store and retrieve type information for selected class
        hierarchies. This is useful for reconstructing a concrete class from
        unstructured data.
    """

    def register_hierarchy(self, cls, cls_fn):
        """
        Register a class hierarchy for typed structure/unstructure
        conversion.

        For each class in the hierarchy under (including) ``cls``, this will
        store an additional ``__type__`` attribute (a string) in the object
        dictionary. This ``__type__`` string will be the concrete class of the
        object, and will be used to structure the dictionary back into an
        object of the correct class.

        For this to work, this function needs to know how to convert a
        ``__type__`` string back into a class, for which it used the provided
        ``cls_fn``.

        Args:
            cls (class):

                    The top-level class of the hierarchy to register.
            cls_fn (function):

                    A function mapping type strings to classes. This can be as
                    simple as ``lambda typ: eval(typ)``, if all subclasses of
                    ``cls`` are visible to the module that calls this method.
        Raises:
            TypeError: If ``cls`` is not a class.
        Example:

            If class ``A`` is the base of class ``B``, and
            ``converter.register_hierarchy(A, lambda typ: eval(typ))`` has been
            called, the dictionary ``y = converter.unstructure(x)`` will
            contain a ``__type__`` field that is ``'A'`` if ``x = A()`` and
            ``B`` if ``x = B()``.

            This ``__type__`` field is then used by ``x =
            converter.structure(y, A)`` to recreate the concrete type of ``x``.
        Note:
                This method is used to register a class hierarchy for typed
                structure/unstructure conversion. For each class in the hierarchy
                under (including) ``cls``, this will store an additional
                ``__type__`` attribute (a string) in the object dictionary. This
                ``__type__`` string will be the concrete class of the object, and
                will be used to structure the dictionary back into an object of the
                correct class.

                For this to work, this function needs to know how to convert a
                ``__type__`` string back into a class, for which it used the
                provided ``cls_fn``.
        """

        self.register_unstructure_hook(cls, lambda obj: self.__typed_unstructure(obj))

        self.register_structure_hook(
            cls, lambda obj_data, cls: self.__typed_structure(obj_data, cls, cls_fn)
        )

    def __typed_unstructure(self, obj):
        """
        Unstructure an object, adding a '__type__' field with the class name.

        Args:
            obj (object): The object to unstructure.
        Returns:
            Dict: The unstructured object.
        Examples:
            >>> converter.__typed_unstructure(A())
            {'__type__': 'A'}
        """
        cls = type(obj)
        unstructure_fn = make_dict_unstructure_fn(cls, self)
        return {"__type__": type(obj).__name__, **unstructure_fn(obj)}

    def __typed_structure(self, obj_data, cls, cls_fn):
        """
        Structure an object, using the '__type__' field to determine the class.

        Args:
            obj_data (Dict): The unstructured object.
            cls (class): The class
            cls_fn (function): A function mapping type strings to classes.
        Returns:
            object: The structured object.
        Raises:
            ValueError: If the '__type__' field is missing.
        Examples:
            >>> converter.__typed_structure({'__type__': 'A'}, A, lambda typ: eval(typ
            'A')
        Note:
            This method is used to structure an object, using the '__type__' field
            to determine the class. This is useful for reconstructing a concrete
            class from unstructured data.
        """
        try:
            cls = cls_fn(obj_data["__type__"])
            structure_fn = make_dict_structure_fn(cls, self)
            return structure_fn(obj_data, cls)
        except Exception as e:
            print(
                f"Could not structure object of type {obj_data}. will try unstructured data. attr __type__ can be missing because of old version of the data."
            )
            print(e)
            return obj_data


# The global converter object, to be used by stores to convert objects into
# dictionaries and back.
converter = TypedConverter()

# register all type-specific hooks with this converter
register_hooks(converter)
