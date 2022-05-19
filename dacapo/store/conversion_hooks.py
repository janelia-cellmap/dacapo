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

from pathlib import Path


def register_hierarchy_hooks(converter):
    """Central place to register type hierarchies for conversion."""

    converter.register_hierarchy(TaskConfig, cls_fun)
    converter.register_hierarchy(ArchitectureConfig, cls_fun)
    converter.register_hierarchy(TrainerConfig, cls_fun)
    converter.register_hierarchy(AugmentConfig, cls_fun)
    converter.register_hierarchy(DataSplitConfig, cls_fun)
    converter.register_hierarchy(DatasetConfig, cls_fun)
    converter.register_hierarchy(ArrayConfig, cls_fun)
    converter.register_hierarchy(GraphStoreConfig, cls_fun)
    converter.register_hierarchy(EvaluationScores, cls_fun)
    converter.register_hierarchy(PostProcessorParameters, cls_fun)
    converter.register_hierarchy(StartConfig, cls_fun)


def register_hooks(converter):
    """Central place to register all conversion hooks with the given
    converter."""

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
    """Convert a type string into the corresponding class. The class must be
    visible to this module (hence the star imports at the top)."""
    return eval(typ)
