from dacapo.experiments.architectures import DummyArchitectureConfig, CellposUNetConfig

import pytest


@pytest.fixture()
def dummy_architecture():
    yield DummyArchitectureConfig(
        name="dummy_architecture", num_in_channels=1, num_out_channels=12
    )


@pytest.fixture
def cellpose_architecture():
    yield CellposUNetConfig(
        name="cellpose_architecture",
        input_shape=(216, 216, 216),
        nbase=[1, 12, 24, 48, 96],
        nout=12,
        conv_3D=True
        # nbase=[1, 32, 64, 128, 256], nout = 32, conv_3D = True
    )
