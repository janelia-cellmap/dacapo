import multiprocessing as mp
import os
import yaml

from dacapo.options import Options

import pytest


@pytest.fixture(params=["fork", "spawn"], autouse=True)
def context(monkeypatch):
    ctx = mp.get_context("spawn")
    monkeypatch.setattr(mp, "Queue", ctx.Queue)
    monkeypatch.setattr(mp, "Process", ctx.Process)
    monkeypatch.setattr(mp, "Event", ctx.Event)
    monkeypatch.setattr(mp, "Value", ctx.Value)


@pytest.fixture(autouse=True)
def runs_base_dir(tmpdir):
    options_file = tmpdir / "dacapo.yaml"
    os.environ["DACAPO_OPTIONS_FILE"] = f"{options_file}"

    with open(options_file, "w") as f:
        options_file.write(yaml.safe_dump({"runs_base_dir": f"{tmpdir}"}))

    assert Options.config_file() == options_file
    assert Options.instance().runs_base_dir == tmpdir
