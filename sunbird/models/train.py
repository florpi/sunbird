from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

from sunbird.data import DSDataModule
from sunbird.models import FCN


def fit(args):
    # Setup data
    with open(args.train_test_split_path) as f:
        train_test_split = json.load(f)
    dm = DSDataModule.from_argparse_args(args, train_test_split)
    dm.setup()
    # Setup model
    model_dict_args = vars(args)
    model = FCN(
        n_input=dm.n_input,
        n_output=dm.n_output,
        normalization_dict=dm.normalization_dict,
        normalize_covariance=args.normalize_outputs,
        slice_filters=dm.slice_filters,
        select_filters=dm.select_filters,
        **model_dict_args,
    )
    # Setup trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=8, mode="min")

    logger = pl_loggers.TensorBoardLogger(save_dir=args.model_dir, name=args.run_name)
    checkpoint_dir = Path(logger.experiment.log_dir) / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="{epoch}-{val_loss:.5f}",
    )
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        accelerator="auto",
        callbacks=[early_stop_callback, checkpoint_callback],
    )
    dm.dump_summaries(path=Path(trainer.log_dir) / "summary.json")
    # Train
    trainer.fit(
        model,
        dm,
    )
    # Test
    # trainer.test(datamodule=dm, ckpt_path="best")
    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    # ensure reproducibility.
    # https://pytorch.org/docs/stable/notes/randomness.html
    seed_everything(0)

    parser = ArgumentParser()
    parser = DSDataModule.add_argparse_args(parser)
    parser.add_argument("--model_dir", type=str, default='../../trained_models/')
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--train_test_split_path", type=str, default="../../data/train_test_split.json"
    )
    parser = Trainer.add_argparse_args(parser)
    parser = FCN.add_model_specific_args(parser)
    args = parser.parse_args()
    val_loss = fit(args)
    print("val loss = ", val_loss)
