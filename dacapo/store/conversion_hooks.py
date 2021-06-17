# star imports ensure visibility of concrete classes, so here they are accepted
# flake8: noqa: F405
from dacapo.experiments.architectures import *
from dacapo.experiments.datasets import *
from dacapo.experiments.datasets.dataset_config import DataSourceConfigs
from dacapo.experiments.datasets.keys import *
from dacapo.experiments.tasks import *
from dacapo.experiments.tasks.evaluators import *
from dacapo.experiments.tasks.post_processors import *
from dacapo.experiments.trainers import *
from pathlib import Path

def register_hierarchy_hooks(converter):
    """Central place to register type hierarchies for conversion."""

    converter.register_hierarchy(TaskConfig, cls_fun)
    converter.register_hierarchy(ArchitectureConfig, cls_fun)
    converter.register_hierarchy(TrainerConfig, cls_fun)
    converter.register_hierarchy(ArraySourceConfig, cls_fun)
    converter.register_hierarchy(GraphSourceConfig, cls_fun)
    converter.register_hierarchy(EvaluationScores, cls_fun)
    converter.register_hierarchy(PostProcessorParameters, cls_fun)

def register_hooks(converter):
    """Central place to register all conversion hooks with the given
    converter."""

    #########################
    # DaCapo specific hooks #
    #########################

    # class hierarchies:
    register_hierarchy_hooks(converter)

    # data source dictionaries:
    converter.register_structure_hook(
        DataSourceConfigs,
        lambda obj, cls: {
            converter.structure(key, DataKey):
                converter.structure(
                    value,
                    ArraySourceConfig
                    if isinstance(key, ArrayKey)
                    else GraphSourceConfig)
            for key, value in obj.items()
        })

    # data key enums:
    converter.register_unstructure_hook(
        DataKey,
        lambda obj: type(obj).__name__ + '::' + obj.value,
    )
    converter.register_structure_hook(
        DataKey,
        lambda obj, _: eval(obj.split('::')[0])(obj.split('::')[1]),
    )

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

def cls_fun(typ):
    """Convert a type string into the corresponding class. The class must be
    visible to this module (hence the star imports at the top)."""
    return eval(typ)
