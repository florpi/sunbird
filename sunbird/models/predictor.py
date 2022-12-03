
import torch
import numpy as np
import yaml
from typing import List
from pathlib import Path
import pytorch_lightning as pl
from sunbird.models import FCN
from sunbird.data.data import load_summary_training

TRAIN_DIR = Path(__file__).parent

class Predictor(pl.LightningModule):
    def __init__(
        self, 
        nn_model, 
        summary_stats, 
        s,
        normalize_inputs: bool = True,
        normalize: bool = False,
        standarize: bool = False,
        apply_s2: bool = False,
        multipoles: List[int] = [0,1],
    ):
        super().__init__()
        self.nn_model = nn_model
        self.summary_stats = summary_stats
        self.s = s
        self.multipoles = multipoles
        self.normalize = normalize
        self.standarize = standarize
        self.apply_s2 = apply_s2
        self.normalize_inputs = normalize_inputs

    @classmethod
    def from_folder(cls, path_to_model: Path):
        path_to_model = Path(path_to_model)
        with open(path_to_model / 'hparams.yaml') as f:
            config = yaml.safe_load(f)
        nn_model = FCN.from_folder(path_to_model)
        nn_model.eval()
        data_dir = TRAIN_DIR.parent.parent / 'data/different_hods/training/ds/gaussian/'
        s = np.load(data_dir / 's.npy')
        s = np.array(list(s) + list(s))
        summary_stats = load_summary_training(
            data_dir= data_dir,
            statistic=config['statistic'],
            s=s,
            apply_s2=config['apply_s2'],
        )
        for k, v in summary_stats.items():
            summary_stats[k] = torch.tensor(v, dtype=torch.float32, requires_grad=False)
        return cls(
            nn_model=nn_model,
            summary_stats=summary_stats,
            s=torch.tensor(s, dtype=torch.float32, requires_grad=False,),
            normalize_inputs=config['normalize_inputs'],
            normalize=config['normalize'],
            standarize=config['standarize'],
            apply_s2=config['apply_s2'],
        )

    def forward(self, inputs: torch.tensor):
        if self.normalize_inputs:
            inputs = (inputs - self.summary_stats['x_min']) / (self.summary_stats['x_max'] - self.summary_stats['x_min'])
        prediction = self.nn_model(inputs)
        if self.normalize:
            prediction = prediction*(
                self.summary_stats['y_max'] - self.summary_stats['y_min']
            ) + self.summary_stats['y_min']
        elif self.standarize:
            prediction = prediction*self.summary_stats['y_std'] + self.summary_stats['y_mean']
        if self.apply_s2:
            prediction = prediction/self.s**2
        return prediction


            
