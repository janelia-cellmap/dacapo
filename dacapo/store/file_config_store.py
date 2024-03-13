from .config_store import ConfigStore, DuplicateNameError
from .converter import converter
from dacapo.experiments import RunConfig
from dacapo.experiments.architectures import ArchitectureConfig
from dacapo.experiments.datasplits import DataSplitConfig
from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
from dacapo.experiments.tasks import TaskConfig
from dacapo.experiments.trainers import TrainerConfig

import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class FileConfigStore(ConfigStore):
    """A Local File based store for configurations. Used to store and retrieve
    configurations for runs, tasks, architectures, trainers, and datasplits.
    """

    def __init__(self, path):
        print("Creating FileConfigStore:\n\tpath: %s" % path)

        self.path = Path(path)

        self.__open_collections()
        self.__init_db()

    def store_run_config(self, run_config, ignore=None):
        run_doc = converter.unstructure(run_config)
        self.__save_insert(self.runs, run_doc, ignore)

    def retrieve_run_config(self, run_name):
        run_doc = self.__load(self.runs, run_name)
        return converter.structure(run_doc, RunConfig)

    def retrieve_run_config_names(self):
        return [f.name[:-5] for f in self.runs.iterdir()]

    def store_task_config(self, task_config, ignore=None):
        task_doc = converter.unstructure(task_config)
        self.__save_insert(self.tasks, task_doc, ignore)

    def retrieve_task_config(self, task_name):
        task_doc = self.__load(self.tasks, task_name)
        return converter.structure(task_doc, TaskConfig)

    def retrieve_task_config_names(self):
        return [f.name[:-5] for f in self.tasks.iterdir()]

    def store_architecture_config(self, architecture_config, ignore=None):
        architecture_doc = converter.unstructure(architecture_config)
        self.__save_insert(self.architectures, architecture_doc, ignore)

    def retrieve_architecture_config(self, architecture_name):
        architecture_doc = self.__load(self.architectures, architecture_name)
        return converter.structure(architecture_doc, ArchitectureConfig)

    def retrieve_architecture_config_names(self):
        return [f.name[:-5] for f in self.architectures.iterdir()]

    def store_trainer_config(self, trainer_config, ignore=None):
        trainer_doc = converter.unstructure(trainer_config)
        self.__save_insert(self.trainers, trainer_doc, ignore)

    def retrieve_trainer_config(self, trainer_name):
        trainer_doc = self.__load(self.trainers, trainer_name)
        return converter.structure(trainer_doc, TrainerConfig)

    def retrieve_trainer_config_names(self):
        return [f.name[:-5] for f in self.trainers.iterdir()]

    def store_datasplit_config(self, datasplit_config, ignore=None):
        datasplit_doc = converter.unstructure(datasplit_config)
        self.__save_insert(self.datasplits, datasplit_doc, ignore)

    def retrieve_datasplit_config(self, datasplit_name):
        datasplit_doc = self.__load(self.datasplits, datasplit_name)
        return converter.structure(datasplit_doc, DataSplitConfig)

    def retrieve_datasplit_config_names(self):
        return [f.name[:-5] for f in self.datasplits.iterdir()]

    def store_array_config(self, array_config, ignore=None):
        array_doc = converter.unstructure(array_config)
        self.__save_insert(self.arrays, array_doc, ignore)

    def retrieve_array_config(self, array_name):
        array_doc = self.__load(self.arrays, array_name)
        return converter.structure(array_doc, ArrayConfig)

    def retrieve_array_config_names(self):
        return [f.name[:-5] for f in self.arrays.iterdir()]

    def __save_insert(self, collection, data, ignore=None):
        name = data["name"]

        file_store = collection / f"{name}.yaml"
        if not file_store.exists():
            with file_store.open("w") as f:
                yaml.dump(dict(data), f)

        else:
            with file_store.open("r") as f:
                existing = yaml.full_load(f)

            if not self.__same_doc(existing, data, ignore):
                raise DuplicateNameError(
                    f"Data for {name} does not match already stored "
                    f"entry. Found\n\n{existing}\n\nin DB, but was "
                    f"given\n\n{data}"
                )

    def __load(self, collection, name):
        file_store = collection / f"{name}.yaml"
        if file_store.exists():
            with file_store.open("r") as f:
                return yaml.full_load(f)
        else:
            raise ValueError(f"No config with name: {name} in collection: {collection}")

    def __same_doc(self, a, b, ignore=None):
        if ignore:
            a = dict(a)
            b = dict(b)
            for key in ignore:
                if key in a:
                    del a[key]
                if key in b:
                    del b[key]

        return a == b

    def __init_db(self):
        # no indexing for filesystem
        # please only use this config store for debugging
        pass

    def __open_collections(self):
        self.users.mkdir(exist_ok=True, parents=True)
        self.runs.mkdir(exist_ok=True, parents=True)
        self.tasks.mkdir(exist_ok=True, parents=True)
        self.datasplits.mkdir(exist_ok=True, parents=True)
        self.arrays.mkdir(exist_ok=True, parents=True)
        self.architectures.mkdir(exist_ok=True, parents=True)
        self.trainers.mkdir(exist_ok=True, parents=True)

    @property
    def users(self) -> Path:
        return self.path / "users"

    @property
    def runs(self) -> Path:
        return self.path / "runs"

    @property
    def tasks(self) -> Path:
        return self.path / "tasks"

    @property
    def datasplits(self) -> Path:
        return self.path / "datasplits"

    @property
    def arrays(self) -> Path:
        return self.path / "arrays"

    @property
    def architectures(self) -> Path:
        return self.path / "architectures"

    @property
    def trainers(self) -> Path:
        return self.path / "trainers"

    @property
    def datasets(self) -> Path:
        return self.path / "datasets"

    def delete_config(self, database: Path, config_name: str) -> None:
        (database / f"{config_name}.yaml").unlink()
