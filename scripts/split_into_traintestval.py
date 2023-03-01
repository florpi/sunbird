import numpy as np 
import json
import random

if __name__ == "__main__":
    reduced_grid = False
    if reduced_grid:
        derivative_grid = [
            100, 101, 102, 103, 104,
            105, 112, 113, 117, 118,
            119, 120, 125, 126,
        ]
        lhc_grid = list(range(130, 147)) + [116]
    else:
        derivative_grid = list(range(100, 127))
        lhc_grid = list(range(130, 182)) 

    percent_val = 0.1
    random.shuffle(lhc_grid)
    n_samples = len(lhc_grid)
    n_val = int(np.floor(percent_val * n_samples))
    val_idx = lhc_grid[:n_val]
    train_idx = list(lhc_grid[n_val:]) + derivative_grid

    split_dict = {
        'train': [int(idx) for idx in train_idx],
        'val': [int(idx) for idx in val_idx],
        'test': list(range(0, 5)) + list(range(13, 14)),
    }
    filename = 'train_test_split_reduced.json' if reduced_grid else 'train_test_split.json'
    with open(filename, 'w') as f:
        json.dump(split_dict, f)
