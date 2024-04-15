# star imports ensure visibility of concrete classes, so here they are accepted
# flake8: noqa: F405
from dacapo.experiments.architectures import *
from dacapo.experiments.datasplits import *
from dacapo.experiments.datasplits.datasets import *
from dacapo.experiments.datasplits.datasets.arrays import *
from dacapo.experiments.datasplits.datasets.graphstores import *
from dacapo.experiments.tasks import *
from dacapo.experiments.tasks.evaluators import *
from dacapo.experiments.tasks.post_processors import *
from dacapo.experiments.trainers import *
from dacapo.experiments.trainers.gp_augments import *
from dacapo.experiments.starts import *

from funlib.geometry import Coordinate, Roi

from upath import UPath as Path


def register_hierarchy_hooks(converter):
    """
    Central place to register type hierarchies for conversion.

    Args:
        converter (Converter): The converter to register the hooks with.
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

    converter.register_hierarchy(TaskConfig, cls_fun)
    converter.register_hierarchy(StartConfig, cls_fun)
    converter.register_hierarchy(ArchitectureConfig, cls_fun)
    converter.register_hierarchy(TrainerConfig, cls_fun)
    converter.register_hierarchy(AugmentConfig, cls_fun)
    converter.register_hierarchy(DataSplitConfig, cls_fun)
    converter.register_hierarchy(DatasetConfig, cls_fun)
    converter.register_hierarchy(ArrayConfig, cls_fun)
    converter.register_hierarchy(GraphStoreConfig, cls_fun)
    converter.register_hierarchy(EvaluationScores, cls_fun)
    converter.register_hierarchy(PostProcessorParameters, cls_fun)


def register_hooks(converter):
    """
    Central place to register all conversion hooks with the given
    converter.

    Args:
        converter (Converter): The converter to register the hooks with.
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

    #########################
    # DaCapo specific hooks #
    #########################

    # class hierarchies:
    register_hierarchy_hooks(converter)

    #################
    # general hooks #
    #################

    # path to string and back
    converter.register_unstructure_hook(
        Path,
        lambda o: str(o),
    )
    converter.register_structure_hook(
        Path,
        lambda o, _: Path(o),
    )

    # Coordinate to tuple and back
    converter.register_unstructure_hook(
        Coordinate,
        lambda o: tuple(o),
    )
    converter.register_structure_hook(
        Coordinate,
        lambda o, _: Coordinate(o),
    )

    # Roi to coordinate tuple and back
    converter.register_unstructure_hook(
        Roi,
        lambda o: (converter.unstructure(o.offset), converter.unstructure(o.shape)),
    )
    converter.register_structure_hook(
        Roi,
        lambda o, _: Roi(*o),
    )


def cls_fun(typ):
    """
    Convert a type string into the corresponding class. The class must be
    visible to this module (hence the star imports at the top).

    Args:
        typ (str): The type string to convert.
    Returns:
        class: The class corresponding to the type string.
    Raises:
        NameError: If the class is not visible to this module.
    Example:
        ``cls_fun('TaskConfig')`` will return the class ``TaskConfig``.
    Note:
        This function is used to convert a type string back into a class. It is
        used in conjunction with the ``register_hierarchy`` function to
        register a class hierarchy for typed structure/unstructure conversion.
    """
    return eval(typ)
