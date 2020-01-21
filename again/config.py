from .losses import *  # noqa
from .models import *  # noqa
from .optimizers import *  # noqa
from .prediction_types import *  # noqa
from .task_types import *  # noqa
from pathlib import Path
import configparser
import hashlib
import os


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

        d = {'id': self.id}

        for key, item in self._config[self._default_section].items():

            item = eval(item)
            if type(item) == type:
                d[key] = item.__name__
            else:
                d[key] = item

        for section in self._config:

            if section in [self._default_section, 'DEFAULT']:
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
            return ConfigWrapper(
                self.config_file,
                attr,
                id_=self.id + '::' + attr)
        else:
            raise AttributeError(
                f"configuration {self.id} is missing a value for {attr}")

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)

    def __repr__(self):
        return self.id


class Data(ConfigWrapper):

    def __init__(self, config_file):
        super(Data, self).__init__(config_file, 'data')
        self.filename = Path(self.id).parent / self.filename

class Task(ConfigWrapper):

    def __init__(self, config_file):
        super(Task, self).__init__(config_file, 'task')
        try:
            self.data = Data(self.data + '.conf')
        except IOError:
            raise IOError(
                f"Config file {self.data + '.conf'} does not exist "
                f"(referenced in task {self})")


class Model(ConfigWrapper):

    def __init__(self, config_file):
        super(Model, self).__init__(config_file, 'model')


class Optimizer(ConfigWrapper):

    def __init__(self, config_file):
        super(Optimizer, self).__init__(config_file, 'optimizer')


class Run:

    def __init__(self, task, model, optimizer):

        self.task = task
        self.model = model
        self.optimizer = optimizer

        run_hash = hashlib.md5()
        run_hash.update(self.task.id.encode())
        run_hash.update(self.model.id.encode())
        run_hash.update(self.optimizer.id.encode())
        self.id = run_hash.hexdigest()

    def to_dict(self):

        return {
            'id': self.id,
            'task': self.task.id,
            'model': self.model.id,
            'optimizer': self.optimizer.id
        }

    def __repr__(self):
        return f"{self.task} with {self.model}, using {self.optimizer}"


def find_tasks(basedir):
    return [Task(f) for f in find_configs(basedir)]


def find_models(basedir):
    return [Model(f) for f in find_configs(basedir)]


def find_optimizers(basedir):
    return [Optimizer(f) for f in find_configs(basedir)]


def enumerate(tasks, models, optimizers):

    configs = []
    for task in tasks:
        for model in models:
            for optimizer in optimizers:
                configs.append(Run(task, model, optimizer))
    return configs


def find_configs(basedir):
    return Path(basedir).rglob('*.conf')


def parse_config_file(filename):

    if not Path(filename).is_file():
        raise IOError(f"Can not find config file {filename}")

    config = configparser.ConfigParser()
    config.read(filename)
    return config


def name_from_config_file(path):

    # remove ".conf" extension
    return os.path.splitext(path)[0]
