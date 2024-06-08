import joblib
import optuna


def objective(trial):
    same_n_hidden = False
    learning_rate = trial.suggest_float("learning_rate",1.0e-3,0.01,)
    weight_decay  = trial.suggest_float("weight_decay", 1.0e-5, 0.001)
    n_layers      = trial.suggest_int("n_layers", 1, 10)
    if same_n_hidden:
        n_hidden  = [trial.suggest_int("n_hidden", 200, 1024)] * n_layers
    else:
        n_hidden  = [trial.suggest_int(f"n_hidden_{layer}", 200, 1024) for layer in range(n_layers)]
        
    dropout_rate  = trial.suggest_float("dropout_rate", 0.0, 0.15)
    args = [n_hidden, dropout_rate, learning_rate, weight_decay]
    return fit(args)


if __name__ == "__main__":
    
    n_trials = 100
    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)

    print(study.best_params)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    joblib.dump(
        study,
        "/data/wliu/home/DESIacm/EMC/c000_ph000/Emulator/first_study.pkl",
    )
