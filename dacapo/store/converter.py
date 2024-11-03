from cattr import Converter
from cattr.gen import make_dict_unstructure_fn, make_dict_structure_fn
from .conversion_hooks import register_hooks


class TypedConverter(Converter):
    def register_hierarchy(self, cls, cls_fn):
        self.register_unstructure_hook(cls, lambda obj: self.__typed_unstructure(obj))

        self.register_structure_hook(
            cls, lambda obj_data, cls: self.__typed_structure(obj_data, cls, cls_fn)
        )

    def __typed_unstructure(self, obj):
        cls = type(obj)
        unstructure_fn = make_dict_unstructure_fn(cls, self)
        return {"__type__": type(obj).__name__, **unstructure_fn(obj)}

    def __typed_structure(self, obj_data, cls, cls_fn):
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
