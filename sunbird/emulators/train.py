from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

from sunbird.data import AbacusDataModule
from sunbird.emulators import FCN


def fit(args):
    # Setup data
    with open(args.train_test_split_path) as f:
        train_test_split = json.load(f)
    dm = AbacusDataModule.from_argparse_args(args, train_test_split)
    dm.setup()
    # Setup model
    model_dict_args = vars(args)
    filtered_model_dict_args = {k:v for k,v in model_dict_args.items() if k != 'output_transforms'}
    model = FCN(
        n_input=dm.n_input,
        n_output=dm.n_output,
        output_transforms=dm.output_transforms,
        slice_filters=dm.slice_filters,
        select_filters=dm.select_filters,
        **filtered_model_dict_args,
    )
    # Setup trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=16, mode="min")

    logger = pl_loggers.TensorBoardLogger(save_dir=args.model_dir, name=args.run_name)
    checkpoint_dir = Path(logger.experiment.log_dir) / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        every_n_epochs=1,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="{epoch}-{val_loss:.5f}",
    )
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        accelerator="auto",
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=0.5,
    )
    dm.store_transforms(path=Path(trainer.log_dir) / "transforms")
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
    parser = AbacusDataModule.add_argparse_args(parser)
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
