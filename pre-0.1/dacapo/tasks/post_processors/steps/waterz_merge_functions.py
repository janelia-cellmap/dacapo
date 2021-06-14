from enum import Enum

from dacapo.converter import converter


class MergeFunction(Enum):
    HIST_QUANT_10 = "hist_quant_10"
    HIST_QUANT_10_INITMAX = "hist_quant_10_initmax"
    HIST_QUANT_25 = "hist_quant_25"
    HIST_QUANT_25_INITMAX = "hist_quant_25_initmax"
    HIST_QUANT_50 = "hist_quant_50"
    HIST_QUANT_50_INITMAX = "hist_quant_50_initmax"
    HIST_QUANT_75 = "hist_quant_75"
    HIST_QUANT_75_INITMAX = "hist_quant_75_initmax"
    HIST_QUANT_90 = "hist_quant_90"
    HIST_QUANT_90_INITMAX = "hist_quant_90_initmax"
    MEAN = "mean"


converter.register_unstructure_hook(
    MergeFunction,
    lambda o: {"value": o.value},
)
converter.register_structure_hook(
    MergeFunction,
    lambda o, _: MergeFunction(o["value"]),
)