import numpy as np
from sunbird.emulators import FCN, train
from sunbird.data import ArrayDataModule
# Please write your own function to load training data
def fit(args):

    n_hidden, dropout_rate, learning_rate, weight_decay = args
    lhc_train_x, lhc_train_y = read_trainset()
    
    train_y_mean = np.mean(lhc_train_y, axis=0)
    train_y_std  = np.std(lhc_train_y, axis=0)
    train_x_mean = np.mean(lhc_train_x, axis=0)
    train_x_std  = np.std(lhc_train_x, axis=0)       
    
    data = ArrayDataModule(x=lhc_train_x,
                        y=lhc_train_y, 
                        val_fraction=0.2, batch_size=64,
                        num_workers=64)
    data.setup("train")
    
    model = FCN(
            n_input=data.n_input,
            n_output=data.n_output,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate, 
            learning_rate=learning_rate,
            scheduler_patience=30,
            scheduler_factor=0.5,
            scheduler_threshold=1.e-6,
            weight_decay=weight_decay,
            act_fn='learned_sigmoid',
            # act_fn='SiLU',
            loss='rmse',
            training=True,
            mean_input=torch.Tensor(train_x_mean),
            std_input=torch.Tensor(train_x_std),
            mean_output=torch.Tensor(train_y_mean),
            std_output=torch.Tensor(train_y_std),
            standarize_input=True,
            standarize_output=True,
        )
    
    val_loss, model, early_stop_callback = train.fit(
        data=data, model=model,
        model_dir=f'/data/wliu/home/DESIacm/EMC/c000_ph000/Emulator/',
    )
    
    return val_loss
