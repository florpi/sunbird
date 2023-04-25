from pathlib import Path
import yaml
from sunbird.summaries.base import BaseSummary


DEFAULT_PATH = Path(__file__).parent.parent.parent / "trained_models/"
DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data/"


class DensitySplit(BaseSummary):
    def __init__(
        self,
        correlation: str,
        dataset: str,
        path_to_models: Path = DEFAULT_PATH,
        path_to_data: Path = DEFAULT_DATA_PATH,
        flax: bool = False,
    ):
        path_to_model = path_to_models / f"best/ds_{correlation}_{dataset}"
        model, flax_params = self.load_model(
            path_to_model=path_to_model,
            flax=flax,
        )
        input_transforms, output_transforms = self.load_transforms(
            path_to_model=path_to_model
        )
        with open(path_to_model / "hparams.yaml") as f:
            config = yaml.safe_load(f)
        coordinates = self.load_coordinates(config=config, path_to_data=path_to_data)
        super().__init__(
            model=model,
            coordinates=coordinates,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
            flax_params=flax_params,
        )


class DensitySplitCross(DensitySplit):
    def __init__(
        self,
        dataset: str = "boss_wideprior",
        path_to_models: Path = DEFAULT_PATH,
        path_to_data: Path = DEFAULT_DATA_PATH,
        flax: bool = False,
    ):
        super().__init__(
            correlation="cross",
            dataset=dataset,
            path_to_models=path_to_models,
            path_to_data=path_to_data,
            flax=flax,
        )


class DensitySplitAuto(DensitySplit):
    def __init__(
        self,
        dataset: str = "boss_wideprior",
        path_to_models: Path = DEFAULT_PATH,
        path_to_data: Path = DEFAULT_DATA_PATH,
        flax: bool = False,
    ):
        super().__init__(
            correlation="auto",
            dataset=dataset,
            path_to_models=path_to_models,
            path_to_data=path_to_data,
            flax=flax,
        )
