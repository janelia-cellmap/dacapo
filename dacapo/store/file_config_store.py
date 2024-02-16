"""
This module is for the File Config Store class, which is used to create file configuration objects. Methods for 
storing and retrieving configurations for runs, tasks, architectures, trainers, and data splits are included.

Attributes:
    ConfigStore (object): The ConfigStore class provides a base for all the other config stores.
    DuplicateNameError (error): An error to raise when a duplicate name is detected.
    converter (function): A function used to convert between structured and unstructured data.
    RunConfig (class): A class for creating run configuration.
    ArchitectureConfig (class): A class for creating architecture configuration.
    DataSplitConfig (class): A class for creating data split configuration.
    ArrayConfig  (class): A class for creating array configuration.
    TaskConfig (class): A class for creating task configuration.
    TrainerConfig (class): A class for creating trainer configuration.
    logging (module): A module provides functions for logging.
    toml (module): A module for handling TOML files.
    Path (function): A function to create the filesystem path in pathlib format.
    queryset (object): An object used to store the queryset
"""

class FileConfigStore(ConfigStore):
    """
    A class which is used to create file configuration store objects. FileConfigStore helps in storing and 
    retrieving configurations for runs, tasks, architectures, trainers, and data splits, arrays.

    Methods:

    __init__:
        Initializes the FileConfigStore object.
        Args:
            path : Path to the configuration file in the local file system.

    store_run_config:
        Stores the run configuration.
        Args:
            run_config : Configuration to be stored.

    retrieve_run_config:
        Retrieves the run configuration.
        Args:
            run_name : Name of the run configuration to be retrieved.

    retrieve_run_config_names:
        Retrieves the names of all run configurations.

    store_task_config:
        Stores the task configuration.
        Args:
            task_config : Configuration to be stored.

    retrieve_task_config:
        Retrieves the task configuration.
        Args:
            task_name : Name of the task configuration to be retrieved.

    retrieve_task_config_names:
        Retrieves the names of all task configurations.
        
    store_architecture_config:
        Stores the architecture configuration.
        Args:
            architecture_config : Configuration to be stored.

    retrieve_architecture_config:
        Retrieves the architecture configuration.
        Args:
            architecture_name : Name of the architecture configuration to be retrieved.

    retrieve_architecture_config_names:
        Retrieves the names of all architecture configurations.

    store_trainer_config:
        Stores the trainer configuration.
        Args:
            trainer_config : Configuration to be stored.

    retrieve_trainer_config:
        Retrieves the trainer configuration.
        Args:
            trainer_name : Name of the trainer configuration to be retrieved.

    retrieve_trainer_config_names:
        Retrieves the names of all trainer configurations.

    store_datasplit_config:
        Stores the data split configuration.
        Args:
            datasplit_config : Configuration to be stored.

    retrieve_datasplit_config:
        Retrieves the data split configuration.
        Args:
            datasplit_name : Name of the data split configuration to be retrieved.

    retrieve_datasplit_config_names:
        Retrieves the names of all data split configurations.

    store_array_config:
        Stores the array configuration.
        Args:
            array_config : Configuration to be stored.

    retrieve_array_config:
        Retrieves the array configuration.
        Args:
            array_name : Name of the array configuration to be retrieved.

    retrieve_array_config_names:
        Retrieves the names of all array configurations.

    __save_insert:
        Saves and inserts the configuration.
        Args:
            collection: The array whereconfigs are being stored.
            data: The data being stored.
            ignore: The data not considered while checking duplicates.

    __load:
        Loads the configuration.
        Args:
            collection: The array from where configs are being retrieved.
            name: Name of the configuration to be retrieved.

    __same_doc:
        Compares two documents.
        Args:
            a: The first document.
            b: The second document.
            ignore: The data not considered while comparing.

    __init_db:
        Initializes the database. This note is important for debugging purposes.

    __open_collections:
        Opens the collections of configuration data.

    users:
        Returns the path to the 'users' configuration files.

    runs:
        Returns the path to the 'runs' configuration files.

    tasks:
        Returns the path to the 'tasks' configuration files.

    datasplits:
        Returns the path to the 'datasplits' configuration files.

    arrays:
        Returns the path to the 'arrays' configuration files.

    architectures:
        Returns the path to the 'architectures' configuration files.

    trainers:
        Returns the path to the 'trainers' configuration files.

    datasets:
        Returns the path to the 'datasets' configuration files.

    delete_config:
        Deletes a specific configuration.
        Args:
            database: The path to the configuration database.
            config_name: The name of the configuration to be deleted.
    """
