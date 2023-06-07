from pathlib import Path
from typing import Optional
from sunbird.summaries.base import BaseSummaryFolder

DEFAULT_PATH = Path(__file__).parent.parent.parent / "trained_models/best/"
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data/"


class Voids(BaseSummaryFolder):
    def __init__(
        self,
        dataset: str = "bossprior",
        loss: str = "mae",
        n_hod_realizations: Optional[int] = None,
        suffix: Optional[str] = None,
        path_to_models: Path = DEFAULT_PATH,
        path_to_data: Path = DEFAULT_DATA_PATH,
        flax: bool = False,
        **kwargs,
    ):
        super().__init__(
            statistic="voids",
            loss=loss,
            n_hod_realizations=n_hod_realizations,
            suffix=suffix,
            dataset=dataset,
            path_to_models=path_to_models,
            path_to_data=path_to_data,
            flax=flax,
        )
