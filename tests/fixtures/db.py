from dacapo import Options

import pymongo
import pytest

import os
from upath import UPath as Path
import yaml


def mongo_db_available():
    options = Options.instance()
    client = pymongo.MongoClient(
        host=options.mongo_db_host,
        serverSelectionTimeoutMS=1000,
        socketTimeoutMS=1000,
    )
    try:
        client.admin.command("ping")
        return True
    except pymongo.errors.ConnectionFailure:
        return False


@pytest.fixture(
    params=(
        "files",
        pytest.param(
            "mongo",
            marks=pytest.mark.skipif(
                not mongo_db_available(), reason="MongoDB not available!"
            ),
        ),
    )
)
def options(request, tmp_path):
    # read the options from the users config file locally
    options = Options.instance(
        type=request.param,
        runs_base_dir="tests",
        mongo_db_name="dacapo_tests",
        compute_context={"type": "LocalTorch", "config": {"device": "cpu"}},
    )

    # change to a temporary directory for this test only
    old_dir = os.getcwd()
    os.chdir(tmp_path)

    # write the dacapo config in the current temporary directory. Now options
    # will be read from this file instead of the users config file letting
    # us test different configurations
    config_file = Path(tmp_path / "dacapo.yaml")
    with open(config_file, "w") as f:
        yaml.safe_dump(options.serialize(), f)
    os.environ["OPTIONS_FILE"] = str(config_file)

    # yield the options
    yield options

    # cleanup
    if request.param == "mongo":
        client = pymongo.MongoClient(host=options.mongo_db_host)
        client.drop_database("dacapo_tests")

    # reset working directory
    os.chdir(old_dir)
