from pathlib import Path
from sunbird.summaries.base import BaseSummary


DEFAULT_PATH = Path(__file__).parent.parent.parent / "trained_models/best/"
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data/"

DensitySplitCross = BaseSummary.from_folder(
    path_to_model = DEFAULT_PATH / "ds_cross/",
)
DensitySplitAuto = BaseSummary.from_folder(
    path_to_model = DEFAULT_PATH / "ds_auto/",
)


