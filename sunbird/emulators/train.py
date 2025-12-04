import torch
import wandb
import logging
import numpy as np
from pathlib import Path
# from warnings import deprecated # Available only in Python 3.13+
from deprecation import deprecated
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from sunbird.emulators import FCN
from sunbird.data import ArrayDataModule

class FCNTrainer(Trainer):
    """
    Trainer class for Fully Connected Neural Network (FCN) models using PyTorch Lightning.
    """
    def __init__(self, callbacks: list = None, logger: str = None, log_dir: str = None, **kwargs):
        """
        Initialize the FCNTrainer with specified callbacks and logger.
        
        Parameters
        ----------
        callbacks : list | None
            List of callbacks to use during training. If None, default callbacks will be set up.
        logger : str | None
            Type of logger to use ('wandb', 'tensorboard', or None).
        log_dir : str | None
            Directory to save logs.
        **kwargs
            Additional keyword arguments for the Trainer.
        """
        
        # Set up default callbacks if none are provided
        patience = kwargs.pop('early_stop_patience', 30)
        min_delta = kwargs.pop('early_stop_threshold', 1.e-7)
        checkpoint_filename = kwargs.pop('checkpoint_filename', '{epoch:02d}-{step}-{val_loss:.5f}')
        checkpoint_dir = kwargs.pop('checkpoint_dir', None)
        if callbacks is None:
            early_stop_callback = self.early_stop_callback(
                monitor="val_loss", 
                patience=patience, 
                min_delta=min_delta,
            )
            checkpoint_callback = self.checkpoint_callback(
                monitor='val_loss',
                checkpoint_filename=checkpoint_filename,
                checkpoint_dir=checkpoint_dir,
            )
            lr_monitor = LearningRateMonitor(logging_interval='step')
            progress_bar = RichProgressBar()
            
            callbacks = [
                early_stop_callback,
                checkpoint_callback,
                lr_monitor,
                progress_bar,
            ]
            callbacks = [cb for cb in callbacks if cb is not None] # Remove None callbacks
            
        logger = self.get_logger(logger=logger, log_dir=log_dir)
        
        gradient_clip_val = kwargs.pop('gradient_clip_val', 0.5)
        log_every_n_steps = kwargs.pop('log_every_n_steps', 1)
        super().__init__(
            callbacks = callbacks,
            logger = logger,
            gradient_clip_val = gradient_clip_val,
            log_every_n_steps = log_every_n_steps,
            **kwargs
        )
        
    def fit(self, *args, **kwargs):
        """
        Fit the model using the Trainer's fit method and return the best validation loss.
        
        Parameters
        ----------
        *args
            Positional arguments to pass to the Trainer's fit method.
        **kwargs
            Keyword arguments to pass to the Trainer's fit method.
        
        Returns
        -------
        best_val_loss : float
            Best validation loss achieved during training.
        """
        super().fit(*args, **kwargs)
        
        # Retrieve the best validation loss from the EarlyStopping callback
        best_val_loss = None
        for cb in self.callbacks:
            if isinstance(cb, EarlyStopping):
                if cb.monitor == 'val_loss':
                    best_val_loss = cb.best_score.item()
                    break
        if best_val_loss is None: # If no EarlyStopping w/ val_loss monitoring found, get it from any other callback
            best_val_loss = self.callback_metrics.get('val_loss', None)
        if best_val_loss is None: # If no callback with val_loss found, raise a warning
            logging.warning('val_loss not found, returning None. If this is not expected, check your callbacks and metrics.')
        
        return best_val_loss
    
    @staticmethod
    def get_logger(logger: str = None, log_dir: str = None):
        """
        Get the logger instance based on the specified type.
        
        Parameters
        ----------
        logger : str | None
            Logger type. Can be 'wandb', 'tensorboard', or None.
        log_dir : str | None
            Directory to save logs. Required if logger is not None.
        
        Returns
        -------
        logger : Logger | None
            Configured logger instance or None.
        """
        if logger == 'wandb':
            wandb.init()
            logger = WandbLogger(log_model="all", project="sunbird",)
        elif logger == 'tensorboard':
            logger = TensorBoardLogger(log_dir, name="optuna")
        elif logger is None:
            logger = None
        return logger
    
    @staticmethod
    def early_stop_callback(
        monitor: str = "val_loss", 
        patience: int = 30, 
        min_delta: float = 1.e-7,
    ) -> EarlyStopping:
        """
        Sets up an EarlyStopping callback for the trainer.

        Parameters
        ----------
        monitor : str
            Metric to monitor for early stopping.
        patience : int
            Number of epochs with no improvement after which training will be stopped.
        min_delta : float
            Minimum change in the monitored metric to qualify as an improvement.

        Returns
        -------
        EarlyStopping
            Configured EarlyStopping callback.
        """
        esc = EarlyStopping(
            monitor = monitor, 
            patience = patience, 
            min_delta = min_delta,
            mode = "min", 
            verbose = True, 
            check_on_train_epoch_end = True,
        )
        return esc
    
    @staticmethod
    def checkpoint_callback(
        monitor: str = "val_loss",
        checkpoint_filename: str = '{epoch:02d}-{step}-{val_loss:.5f}',
        checkpoint_dir: str = None,
    ) -> ModelCheckpoint:
        """
        Sets up a ModelCheckpoint callback for the trainer.
        
        Parameters
        ----------
        monitor : str
            Metric to monitor for saving checkpoints.
        checkpoint_filename : str
            Filename template for the saved checkpoints.
        checkpoint_dir : str | None
            Directory to save the checkpoints. If None, no checkpoints are saved.
        
        Returns
        -------
        ModelCheckpoint | None
            Configured ModelCheckpoint callback or None if checkpoint_dir is None.
        """
        if checkpoint_dir is None: 
            return None
        
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        mcp = ModelCheckpoint(
            monitor = monitor,
            dirpath = checkpoint_dir,
            filename = checkpoint_filename,
            auto_insert_metric_name = True,
            save_last = 'link',
            mode = 'min',
        )
        return mcp


def train_fcn(
    lhc_x: np.ndarray,
    lhc_y: np.ndarray,
    lhc_x_names: list,
    n_hidden: list,
    dropout_rate: float,
    learning_rate: float,
    weight_decay: float,
    act_fn: str = 'learned_sigmoid',
    loss: str = 'mae',
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    scheduler_threshold: float = 1e-6,
    transform = None,
    val_fraction: float = 0.1,
    batch_size: int = 128,
    checkpoint_dir: str = None,
    checkpoint_filename: str = '{epoch:02d}-{step}-{val_loss:.5f}',
    train_logger: str = None,
    log_dir: str = None,
    return_trainer: bool = False,
    **kwargs,
) -> float|tuple[float, FCNTrainer]:
    """
    Train a Fully Connected Neural Network (FCN) model with the given hyperparameters
    and return the validation loss.
    
    Parameters
    ----------
    lhc_x : np.ndarray
        Input features for training.
    lhc_y : np.ndarray
        Target values for training.
    lhc_x_names : list
        Names of the input features.
    n_hidden : list
        List specifying the number of hidden units in each layer.
    dropout_rate : float
        Dropout rate for regularization.
    learning_rate : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay (L2 regularization) for the optimizer.
    act_fn : str
        Activation function to use in the model.
    loss : str
        Loss function to use for training.
    scheduler_patience : int
        The number of allowed epochs with no improvement after which the learning rate will be reduced. (see torch.optim.lr_scheduler)
    scheduler_factor : float
        Factor by which the learning rate will be reduced (see torch.optim.lr_scheduler)
    scheduler_threshold : float
        Threshold for measuring the new learning rate optimum (see torch.optim.lr_scheduler)
    transform : callable | None
        Data transform to apply to the target values.
    val_fraction : float
        Fraction of data to use for validation.
    batch_size : int
        Batch size to use for each dataloader. Default is 128.
    checkpoint_dir : str | None
        Directory to save model checkpoints.
    checkpoint_filename : str
        Filename template for saving checkpoints.
    train_logger : str | None
        Type of logger to use ('wandb', 'tensorboard', or None).
    log_dir : str | None
        Directory to save logs.
    return_trainer: bool
        Whether to return the trainer instance along with the validation loss.
    **kwargs
        Additional keyword arguments for the FCNTrainer.
    
    Returns
    -------
    val_loss : float
        Validation loss after training.
    trainer : FCNTrainer, optional
        The trained FCNTrainer instance (returned if return_trainer is True).
    """
    logger = logging.getLogger('TrainFCN')
    logger.info(f'Loaded LHC with shape: {lhc_x.shape}, {lhc_y.shape}')

    if transform: 
        logger.info(f'Applying transform: {type(transform).__name__}')
        try: # Handle sunbird.data.transforms
            lhc_y = transform.fit_transform(lhc_y)
        except AttributeError: # Handle sunbird.data.transforms_array
            lhc_y = transform.transform(lhc_y) 

    train_mean = np.mean(lhc_y, axis=0)
    train_std = np.std(lhc_y, axis=0)

    train_mean_x = np.mean(lhc_x, axis=0)
    train_std_x = np.std(lhc_x, axis=0)

    data = ArrayDataModule(
        x=torch.Tensor(lhc_x),
        y=torch.Tensor(lhc_y), 
        val_fraction=val_fraction, 
        batch_size=batch_size,
        num_workers=0, # AVoid spreading data amongst sub-processes
    )
    data.setup()

    model = FCN(
        n_input = data.n_input,
        n_output = data.n_output,
        n_hidden = n_hidden,
        dropout_rate = dropout_rate, 
        learning_rate = learning_rate,
        scheduler_patience = scheduler_patience,
        scheduler_factor = scheduler_factor,
        scheduler_threshold = scheduler_threshold,
        weight_decay = weight_decay,
        act_fn = act_fn,
        loss = loss,
        training = True,
        mean_input = train_mean_x,
        std_input = train_std_x,
        mean_output = train_mean,
        std_output = train_std,
        standardize_input = True,
        standardize_output = True,
        transform_input = None,
        transform_output = transform,
        coordinates = lhc_x_names,
    )
    
    trainer = FCNTrainer(
        logger = train_logger,
        log_dir = log_dir,
        checkpoint_dir = checkpoint_dir,
        checkpoint_filename = checkpoint_filename,
        **kwargs,
    )
    
    val_loss = trainer.fit(
        model=model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )
    
    if return_trainer:
        return val_loss, trainer
    return val_loss


@deprecated("Kept for compatibility, use TrainFCN instead")
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
        # filename='best-model-{epoch:02d}-{val_loss:.5f}',
        # save_top_k=1,
        auto_insert_metric_name=True,
        save_last='link',
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    progress_bar = RichProgressBar()

    seed_everything(42, workers=True)

    if logger == 'wandb':
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
        # devices=1,
        **kwargs
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

#%% Example usage
if __name__ == "__main__":
    from sunbird.utils import setup_logging
    setup_logging(level='info')
    seed_everything(42, workers=True)
    
    # Dummy data for testing
    lhc_x = np.random.rand(1000, 10)
    lhc_y = np.random.rand(1000, 5)
    lhc_x_names = [f'feature_{i}' for i in range(10)]
    
    val_loss, trainer = train_fcn(
        lhc_x=lhc_x,
        lhc_y=lhc_y,
        lhc_x_names=lhc_x_names,
        n_hidden=[64, 64],
        dropout_rate=0.1,
        learning_rate=1e-3,
        weight_decay=1e-5,
        max_epochs=10,
        deterministic=True,
        devices=1,
        return_trainer=True,
        checkpoint_dir='./checkpoints',
        checkpoint_filename='test-{epoch:02d}-{val_loss:.2f}.ckpt',
    )
    print(f'Validation Loss: {val_loss}')