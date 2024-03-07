from .compute_context import ComputeContext

import attr

from typing import Optional
import logging

logger = logging.getLogger(__name__)

@attr.s
class Bsub(ComputeContext):
    queue: str = attr.ib(default="local", metadata={"help_text": "The queue to run on"})
    num_gpus: int = attr.ib(
        default=1,
        metadata={
            "help_text": "The number of gpus to train on. "
            "Currently only 1 gpu can be used."
        },
    )
    num_cpus: int = attr.ib(
        default=5,
        metadata={"help_text": "The number of cpus to use to generate training data."},
    )
    billing: Optional[str] = attr.ib(
        default=None,
        metadata={"help_text": "Project name that will be paying for this Job."},
    )
    # log_dir: Optional[str] = attr.ib(
    #     default="~/logs/dacapo/",
    #     metadata={"help_text": "The directory to store the logs in."},
    # )

    @property
    def device(self):
        if self.num_gpus > 0:
            return "cuda"
        else:
            return "cpu"

    def _wrap_command(self, command):
        full_command =  (
            [
                "bsub",
                "-P",
                "cellmap",
                "-q",
                "gpu_tesla",
                "-n",
                f"5",
                "-gpu",
                f"num=1",
                "-J",
                "dacapoooooooooo",
                "-o"
                f"/groups/cellmap/cellmap/zouinkhim/ml_experiments_v2/validate/logs/v22_peroxisome_funetuning_best_v20_1e4_finetuned_distances_8nm_peroxisome_jrc_mus-livers_peroxisome_8nm_attention-upsample-unet_default_one_label_finetuning_0/train.out",
                "-e",
                f"/groups/cellmap/cellmap/zouinkhim/ml_experiments_v2/validate/logs/v22_peroxisome_funetuning_best_v20_1e4_finetuned_distances_8nm_peroxisome_jrc_mus-livers_peroxisome_8nm_attention-upsample-unet_default_one_label_finetuning_0/train.err",
            ]
            + command
        )
        full_command = [str(c) for c in full_command]
        logger.warning(f"Submitting command: {' '.join(full_command)}")
        return full_command
