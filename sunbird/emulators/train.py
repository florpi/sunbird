from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning import Trainer, seed_everything
import torch


def fit(data, model, early_stop_patience=30, early_stop_threshold=1.e-7, max_epochs=1_000, model_dir=None, log_dir=None, logger='wandb', **kwargs):
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        patience=early_stop_patience, 
        min_delta=early_stop_threshold,
        mode="min", 
        verbose=True, 
        check_on_train_epoch_end=True,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=model_dir,
        filename='{epoch:02d}-{train_loss:.5f}-{val_loss:.5f}',
        save_top_k=1,
        auto_insert_metric_name=False,
        save_last='link',
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    progress_bar = RichProgressBar()

    seed_everything(42, workers=True)

    if logger == 'wandb':
        import wandb
        wandb.init()
        logger = WandbLogger(log_model="all", project="sunbird",)
    elif logger == 'tensorboard':
        logger = TensorBoardLogger(log_dir, name="optuna")
    else:
        logger=None

    trainer = Trainer(
        accelerator="auto",
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor, progress_bar],
        gradient_clip_val=0.5,
        deterministic=True,
        max_epochs=max_epochs,
        logger=logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        # num_sanity_val_steps=0,
        # devices=1,
        **kwargs
    )
    #  # === NEW: run a validation pass BEFORE training to save the "init" checkpoint ===
    # # This will compute val_loss with random weights and trigger ModelCheckpoint once.
    # _ = trainer.validate(
    #     model=model,
    #     dataloaders=data.val_dataloader(),
    #     verbose=False,
    #     ckpt_path=None,
    # )
    trainer.fit(
        model=model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )
    best_val_epoch = early_stop_callback.best_score.item()
    weights_dict = torch.load(
        checkpoint_callback.best_model_path,
        map_location=torch.device('cpu'),
        weights_only=False,
    )
    state_dict = weights_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    return best_val_epoch, model, early_stop_callback