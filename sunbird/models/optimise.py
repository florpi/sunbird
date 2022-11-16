import optuna
from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser, Namespace
from sunbird.data import DSDataModule
from sunbird.models import FCN
from sunbird.models.train import fit


def objective(trial, args):
    lr = trial.suggest_float(
        "learning_rate",
        1.0e-4,
        0.1,
    )
    weight_decay = trial.suggest_float("weight_decay", 1.0e-4, 0.01)
    n_layers = trial.suggest_int("n_layers", 1, 6)
    n_hidden = trial.suggest_int("n_hidden", 32, 1024)
    batch_size = trial.suggest_int(
        "batch_size",
        64,
        512,
    )
    args.learning_rate = lr
    args.weight_decay = weight_decay
    args.n_layers = n_layers
    args.n_hidden = n_hidden
    args.batch_size = batch_size
    return fit(args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = DSDataModule.add_argparse_args(parser)

    parser.add_argument("--model_dir", type=str, default="../../trained_models")
    parser.add_argument("--run_name", type=str, default="optuna")

    parser = Trainer.add_argparse_args(parser)
    parser = FCN.add_model_specific_args(parser)
    args = parser.parse_args()

    #args.run_name = f"{args.run_name}_ds{args.quintile}_m{args.multipole}"
    n_trials = 200
    study = optuna.create_study()
    optimize_objective = lambda trial: objective(trial, args)
    study.optimize(optimize_objective, n_trials=n_trials)

    print(study.best_params)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
