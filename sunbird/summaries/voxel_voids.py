from pathlib import Path
from typing import Optional
from sunbird.summaries.base import BaseSummaryFolder

DEFAULT_PATH = Path(__file__).parent.parent.parent / "trained_models/best/"
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data/"

COSMO_PARAMS = [
    "omega_b",
    "omega_cdm",
    "sigma8_m",
    "n_s",
    "nrun",
    "N_ur",
    "w0_fld",
    "wa_fld",
]
GAL_PARAMS = [
    "logM1",
    "logM_cut",
    "alpha",
    "logsigma",
    "kappa",
]
# GAL_PARAMS = [
#     "logM_cut",
#     "logM1",
#     "sigma",
#     "alpha",
#     "kappa",
# ]


class VoxelVoids(BaseSummaryFolder):
    def __init__(
        self,
        dataset: str = "voidprior",
        loss: str = "mae",
        n_hod_realizations: Optional[int] = None,
        suffix: Optional[str] = None,
        path_to_models: Path = DEFAULT_PATH,
        path_to_data: Path = DEFAULT_DATA_PATH,
        input_names=COSMO_PARAMS+GAL_PARAMS,
        flax: bool = False,
        **kwargs,
    ):
        super().__init__(
            statistic="voxel_voids",
            loss=loss,
            n_hod_realizations=n_hod_realizations,
            suffix=suffix,
            dataset=dataset,
            path_to_models=path_to_models,
            path_to_data=path_to_data,
            flax=flax,
            input_names=input_names,
        )
