import os
import tempfile
from dacapo import Options
from os.path import expanduser


from upath import UPath as Path
import textwrap


def test_no_config():
    # Temporarilly move any dacapo config file
    original_files = {}
    while Options.config_file() is not None:
        original_files[Options.config_file()] = Options.config_file().with_suffix(
            ".bak"
        )
        os.rename(Options.config_file(), Options.config_file().with_suffix(".bak"))

    # Remove the environment variable
    env_dict = dict(os.environ)
    if "DACAPO_OPTIONS_FILE" in env_dict:
        del env_dict["DACAPO_OPTIONS_FILE"]

    # Parse the options
    options = Options.instance()

    # Check the options
    assert isinstance(options.runs_base_dir, Path)
    assert options.mongo_db_host is None
    assert options.mongo_db_name is None

    # Parse the options
    options = Options.instance(
        runs_base_dir="tmp", mongo_db_host="localhost", mongo_db_name="dacapo"
    )

    # Check the options
    assert options.runs_base_dir == Path("tmp")
    assert options.mongo_db_host == "localhost"
    assert options.mongo_db_name == "dacapo"

    # Restore the original config files
    for original, new in original_files.items():
        os.rename(new, original)


# we need to change the working directory because
# dacapo looks for the config file in the working directory
def test_local_config_file():
    # get temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a config file
        config_file = Path(tmpdir, "dacapo.yaml")
        config_file.write_text(
            textwrap.dedent(
                """
                runs_base_dir: /tmp
                mongo_db_host: localhost
                mongo_db_name: dacapo
                """
            )
        )
        os.environ["DACAPO_OPTIONS_FILE"] = str(config_file)

        # Parse the options
        options = Options.instance()

        # Check the options
        assert options.runs_base_dir == Path("/tmp")
        assert options.mongo_db_host == "localhost"
        assert options.mongo_db_name == "dacapo"
        assert Options.config_file() == config_file

        # Parse the options
        options = Options.instance(runs_base_dir="/tmp2")

        # Check the options
        assert options.runs_base_dir == Path("/tmp2")
        assert options.mongo_db_host == "localhost"
        assert options.mongo_db_name == "dacapo"
        assert Options.config_file() == config_file
