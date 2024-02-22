from dacapo import Options


from pathlib import Path
import textwrap


def test_no_config():
    # Make sure the config file does not exist
    config_file = Path("dacapo.yaml")
    if config_file.exists():
        config_file.unlink()

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


# we need to change the working directory because
# dacapo looks for the config file in the working directory
def test_local_config_file():
    # Create a config file
    config_file = Path("dacapo.yaml")
    config_file.write_text(
        textwrap.dedent(
            """
            runs_base_dir: /tmp
            mongo_db_host: localhost
            mongo_db_name: dacapo
            """
        )
    )

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
