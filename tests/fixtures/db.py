from dacapo import Options

import pymongo
import pytest


def mongo_db_available():
    try:
        options = Options.instance()
        client = pymongo.MongoClient(
            host=options.mongo_db_host, serverSelectionTimeoutMS=1000
        )
        Options._instance = None
    except RuntimeError:
        # cannot find a dacapo config file, mongodb is not available
        Options._instance = None
        return False
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
    # TODO: Clean up this fixture. Its a bit clunky to use.
    # Maybe just write the dacapo.yaml file instead of assigning to Options._instance
    kwargs_from_file = {}
    if request.param == "mongo":
        options_from_file = Options.instance()
        kwargs_from_file.update(
            {
                "mongo_db_host": options_from_file.mongo_db_host,
                "mongo_db_name": "dacapo_tests",
            }
        )
    Options._instance = None
    options = Options.instance(
        type=request.param, runs_base_dir=f"{tmp_path}", **kwargs_from_file
    )
    yield options
    if request.param == "mongo":
        client = pymongo.MongoClient(host=options.mongo_db_host)
        client.drop_database("dacapo_tests")
    Options._instance = None
