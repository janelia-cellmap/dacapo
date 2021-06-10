from cattr import Converter
from funlib.geometry import Coordinate


from pathlib import Path

# specific structure and unstructure hooks can be added
# to this converter
converter = Converter()

# general hooks
# path to string and back
converter.register_unstructure_hook(
    Path,
    lambda o: str(o),
)
converter.register_structure_hook(
    Path,
    lambda o, _: Path(o),
)

# coordinate to tuple and back
converter.register_unstructure_hook(
    Coordinate,
    lambda o: tuple(o),
)
converter.register_structure_hook(
    Coordinate,
    lambda o, _: Coordinate(o),
)