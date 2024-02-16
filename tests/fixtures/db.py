from dacapo import Options

import pymongo
import pytest
import attr

import os
from pathlib import Path
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
        type=request.param, runs_base_dir="tests", mongo_db_name="dacapo_tests"
    )

    # change to a temporary directory for this test only
    old_dir = os.getcwd()
    os.chdir(tmp_path)

    # write the dacapo config in the current temporary directory. Now options
    # will be read from this file instead of the users config file letting
    # us test different configurations
    config_file = Path("dacapo.yaml")
    config_file.write_text(
        yaml.safe_dump(
            attr.asdict(
                options,
                value_serializer=lambda inst, field, value: (
                    str(value) if value is not None else None
                ),
            )
        )
    )

    # yield the options
    yield options

    # cleanup
    if request.param == "mongo":
        client = pymongo.MongoClient(host=options.mongo_db_host)
        client.drop_database("dacapo_tests")

    # reset working directory
    os.chdir(old_dir)
