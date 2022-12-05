from pathlib import Path
from sunbird.models.predictor import Predictor
from sunbird.summaries.base import BaseSummary

DEFAULT_PATH = (
    Path(__file__).parent.parent.parent / "trained_models/tpcf_final/version_0/"
)
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data/"
# TODO: Take care of normalization properly, by reading from hparams.yaml


class TPCF(BaseSummary):
    def __init__(
        self,
        path_to_model: Path = DEFAULT_PATH,
        path_to_data: Path = DEFAULT_DATA_PATH,
    ):
        super().__init__(
            path_to_data=path_to_data,
            path_to_model=path_to_model,
        )
        self.model = Predictor.from_folder(path_to_model)

    def forward(self, inputs, select_filters, slice_filters):
        output = self.model(inputs)
        if slice_filters is not None:
            if "s" in slice_filters:
                s_min = slice_filters["s"][0]
                s_max = slice_filters["s"][1]
                output = output[:, (self.model.s > s_min) & (self.model.s < s_max)]
            if "multipoles" in slice_filters:
                m_min = slice_filters["multipoles"][0]
                m_max = slice_filters["multipoles"][1]
                output = output.reshape((2, -1, output.shape[-1] // 2))[m_min:m_max]
        if select_filters is not None:
            if "multipoles" in select_filters:
                multipoles = select_filters["multipoles"]
                output = output.reshape((2, -1, output.shape[-1] // 2))
                output = output[multipoles].reshape(-1)
        return output
