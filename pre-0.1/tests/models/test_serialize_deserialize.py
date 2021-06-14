import pytest

from dacapo.converter import converter
from dacapo.models import Model
from .architecture_fixtures import ARCHITECTURES


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_converter(architecture):
    model = Model(name="test_model", architecture=architecture)
    unstructured = converter.unstructure(model)
    restructured = converter.structure(unstructured, Model)
    assert architecture == restructured.architecture
