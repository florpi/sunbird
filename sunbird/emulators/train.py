from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger 
from lightning import Trainer, seed_everything
import torch


def fit(data, model, **kwargs):
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        patience=kwargs['early_stop_patience'], 
        min_delta=kwargs['early_stop_threshold'],
        mode="min", 
        verbose=True, 
        check_on_train_epoch_end=True,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=kwargs['model_dir'],
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    seed_everything(42, workers=True)

    logger = WandbLogger(log_model="all", project="sunbird")

    trainer = Trainer(
        accelerator="auto",
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        gradient_clip_val=0.5,
        deterministic=True,
        max_epochs=1_000,
        logger=logger,
        check_val_every_n_epoch=1,
    )
    trainer.fit(
        model=model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )
    best_val_epoch = early_stop_callback.best_score.item()
    weights_dict = torch.load(
        checkpoint_callback.best_model_path,
        map_location=torch.device('cpu'),
    )
    state_dict = weights_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    return best_val_epoch, model, early_stop_callback