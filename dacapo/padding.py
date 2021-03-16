from gunpowder import Coordinate

from .converter import converter

from enum import Enum


class PaddingOption(Enum):
    VALID = "valid"
    SAME = "same"
    MINIMAL = "minimal"


converter.register_unstructure_hook(
    PaddingOption,
    lambda o: {"value": o.value},
)
converter.register_structure_hook(
    PaddingOption,
    lambda o, _: PaddingOption(o["value"]),
)


def compute_padding(
    input_roi, output_roi, input_size, output_size, voxel_size, padding
):
    # raise Exception(output_size/voxel_size)
    """
    Compute the necessary padding!

    If padding is "valid", no padding is needed, instead output_roi will shrink.
    - will throw an exception if input_roi - context is empty

    If padding is "same", padding will be added to input_roi s.t. input_roi-context=output_roi

    If padding is "minimal", padding will be added to input_roi s.t. input_roi >= output_size
    """
    def f(a, b):
        if a is None:
            return True
        elif b is None:
            return False
        else:
            return a >= b

    context = (input_size - output_size) / 2
    no_pad_output_roi = output_roi.intersect(input_roi.grow(-context, -context))
    no_pad_input_roi = no_pad_output_roi.grow(context, context)
    if padding == PaddingOption.VALID:
        assert all(
            [f(a, b) for a, b in zip(input_roi.shape, input_size)]
        ), f"input_roi {input_roi} is too small to accomodate model input_size: {input_size}"
        assert all(
            [f(a, b) for a, b in zip(output_roi.shape, output_size)]
        ), f"output_roi {output_roi} is too small to accomodate model output_size: {output_size}"
        return no_pad_input_roi, no_pad_output_roi, Coordinate([0] * input_roi.dims)

    elif padding == PaddingOption.SAME:

        padding = (output_roi.shape - no_pad_output_roi.shape) / 2
        padding = (
            (padding + voxel_size - 1) / voxel_size
        ) * voxel_size

        assert all(
            [f(a, b) for a, b in zip(input_roi.shape + padding * 2, input_size)]
        ), f"{input_roi.shape + padding * 2} < {input_size}"
        assert all(
            [f(a, b) for a, b in zip(output_roi.shape + padding * 2, output_size)]
        ), f"{output_roi.shape + padding * 2} < {output_size}"
        return (
            no_pad_input_roi.grow(padding, padding),
            no_pad_output_roi.grow(padding, padding),
            padding,
        )
    elif padding == PaddingOption.MINIMAL:
        padding = (input_size - no_pad_input_roi.shape) / 2
        padding = Coordinate([max(p, 0) for p in padding])
        padding = (
            (padding + voxel_size - 1) / voxel_size
        ) * voxel_size
        assert all(
            [f(a, b) for a, b in zip(input_roi.shape + padding * 2, input_size)]
        ), f"{input_roi.shape + padding * 2} < {input_size}"
        assert all(
            [f(a, b) for a, b in zip(output_roi.shape + padding * 2, output_size)]
        ), f"{output_roi.shape + padding * 2} < {output_size}"
        return (
            no_pad_input_roi.grow(padding, padding),
            no_pad_output_roi.grow(padding, padding),
            padding,
        )
    else:
        raise Exception(f"Padding mode {padding} not valid")