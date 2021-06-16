# star imports ensure visibility of concrete classes, so here they are accepted
# flake8: noqa: F405
from dacapo.experiments.tasks import *
from dacapo.experiments.architectures import *
from dacapo.experiments.trainers import *
from dacapo.experiments.datasets import *
from pathlib import Path

def register_hierarchy_hooks(converter):
    """Central place to register type hierarchies for conversion."""

    converter.register_hierarchy(TaskConfig, cls_fun)
    converter.register_hierarchy(ArchitectureConfig, cls_fun)
    converter.register_hierarchy(TrainerConfig, cls_fun)
    converter.register_hierarchy(ArraySourceConfig, cls_fun)
    converter.register_hierarchy(GraphSourceConfig, cls_fun)

def register_hooks(converter):
    """Central place to register all conversion hooks with the given
    converter."""

    # DaCapo specific hooks

    register_hierarchy_hooks(converter)

    # general hooks

    # path to string and back
    converter.register_unstructure_hook(
        Path,
        lambda o: str(o),
    )
    converter.register_structure_hook(
        Path,
        lambda o, _: Path(o),
    )

def cls_fun(typ):
    """Convert a type string into the corresponding class. The class must be
    visible to this module (hence the star imports at the top)."""
    return eval(typ)
