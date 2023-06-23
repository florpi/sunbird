import optuna
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from argparse import ArgumentParser
from sunbird.data import AbacusDataModule
from sunbird.emulators import FCN
from sunbird.emulators.train import fit


def objective(
    trial,
    args,
):
    same_n_hidden = False
    lr = trial.suggest_float(
        "learning_rate",
        1.0e-3,
        0.01,
    )
    weight_decay = trial.suggest_float("weight_decay", 1.0e-5, 0.001)
    n_layers = trial.suggest_int("n_layers", 1, 10)
    if same_n_hidden:
        n_hidden = [trial.suggest_int("n_hidden", 200, 1024)] * n_layers
    else:
        n_hidden = [
            trial.suggest_int(f"n_hidden_{layer}", 200, 1024)
            for layer in range(n_layers)
        ]
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.15)
    args.learning_rate = lr
    args.weight_decay = weight_decay
    args.n_hidden = n_hidden
    args.dropout_rate = dropout_rate
    return fit(args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = AbacusDataModule.add_argparse_args(parser)

    parser.add_argument("--model_dir", type=str, default="../../trained_models")
    parser.add_argument("--run_name", type=str, default="optuna")
    parser.add_argument(
        "--train_test_split_path", type=str, default="../../data/train_test_split.json"
    )

    parser = Trainer.add_argparse_args(parser)
    parser = FCN.add_model_specific_args(parser)
    args = parser.parse_args()

    n_trials = 500
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
    joblib.dump(
        study,
        Path(args.model_dir) / f"{args.run_name}/study.pkl",
    )
