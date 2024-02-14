from .evaluation_scores import EvaluationScores
import attr

from typing import Tuple


@attr.s
class IntensitiesEvaluationScores(EvaluationScores):
    criteria: property = ["ssim", "psnr", "nrmse"]

    ssim: float = attr.ib(default=float("nan"))
    psnr: float = attr.ib(default=float("nan"))
    nrmse: float = attr.ib(default=float("nan"))

    @staticmethod
    def higher_is_better(criterion: str) -> bool:
        mapping: dict[str, bool] = {
            "ssim": True,
            "psnr": True,
            "nrmse": False,
        }
        return mapping[criterion]

    @staticmethod
    def bounds(criterion: str) -> Tuple[float, float]:
        mapping: dict[str, tuple] = {
            "ssim": (0, None),
            "psnr": (0, None),
            "nrmse": (0, None),
        }
        return mapping[criterion]

    @staticmethod
    def store_best(criterion: str) -> bool:
        return True
