from pathlib import Path
import configparser
import importlib
import numpy as np
import os

from dacapo.hash import hash_adjective, hash_noun
from dacapo.models import *  # noqa
from dacapo.optimizers import *  # noqa
from dacapo.tasks.losses import *  # noqa
from dacapo.tasks.predictors import *  # noqa
from dacapo.tasks.post_processors import *  # noqa

from dacapo.load_plugins import discovered_plugins

for name in discovered_plugins:
    print(f"Importing {name}")
    globals()[name.split(".")[0]] = __import__(name, globals={"__name__": __name__})


class ConfigWrapper:
    def __init__(self, config_file, default_section, id_=None):

        self.config_file = config_file
        self._config = parse_config_file(config_file)
        self._default_section = default_section
        if id_:
            self.id = id_
        else:
            self.id = name_from_config_file(config_file)

    def to_dict(self):

        d = {"id": self.id}

        for key, item in self._config[self._default_section].items():
            print(f"KEY: {key}, ITEM: {item}")

            item = eval(item)
            if type(item) == type:
                d[key] = item.__name__
            else:
                d[key] = item

        for section in self._config:

            if section in [self._default_section, "DEFAULT"]:
                continue

            d[section] = {}
            for key, item in self._config[section].items():

                item = eval(item)
                if type(item) == type:
                    d[section][key] = item.__name__
                else:
                    d[section][key] = item

        return d

    def __getattr__(self, attr):
        if attr in self._config[self._default_section]:
            return eval(self._config[self._default_section][attr])
        elif attr in self._config:
            return ConfigWrapper(self.config_file, attr, id_=self.id + "::" + attr)
        else:
            raise AttributeError(
                f"configuration {self.id} is missing a value for {attr}"
            )

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)

    def __repr__(self):
        return self.id


class DataConfig(ConfigWrapper):
    def __init__(self, config_file):
        super(DataConfig, self).__init__(config_file, "data")
        self.filename = Path(self.id).parent / self.filename


class TaskConfig(ConfigWrapper):
    def __init__(self, config_file):
        super(TaskConfig, self).__init__(config_file, "task")
        try:
            self.data = DataConfig(self.data + ".conf")
            self.hash = hash_noun(self.id)
        except IOError:
            raise IOError(
                f"Config file {self.data + '.conf'} does not exist "
                f"(referenced in task {self})"
            )


class ModelConfig(ConfigWrapper):
    def __init__(self, config_file):
        super(ModelConfig, self).__init__(config_file, "model")
        self.hash = hash_adjective(self.id)

    def to_dict(self):
        d = super(ModelConfig, self).to_dict()
        return d


class OptimizerConfig(ConfigWrapper):
    def __init__(self, config_file):
        super(OptimizerConfig, self).__init__(config_file, "optimizer")
        self.hash = hash_adjective(self.id)


def find_task_configs(basedir):
    return [TaskConfig(f) for f in find_configs(basedir)]


def find_model_configs(basedir):
    return [ModelConfig(f) for f in find_configs(basedir)]


def find_optimizer_configs(basedir):
    return [OptimizerConfig(f) for f in find_configs(basedir)]


def find_configs(basedir):
    return Path(basedir).rglob("*.conf")


def parse_config_file(filename):

    if not Path(filename).is_file():
        raise IOError(f"Can not find config file {filename}")

    config = configparser.ConfigParser()
    config.read(filename)
    return config


def name_from_config_file(path):

    # remove ".conf" extension
    return os.path.splitext(path)[0]
