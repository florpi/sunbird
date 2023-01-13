import numpy as np 
import json
import random


dataset = 'different_hods'

derivative_grid = list(range(100, 127))
lhc_grid = list(range(130, 182)) + list(range(0, 5)) + list(range(13, 14))
percent_val = 0.1
percent_test = 0.1
random.shuffle(lhc_grid)
n_samples = len(lhc_grid)
n_val = int(np.floor(percent_val * n_samples))
n_test = int(np.floor(percent_test * n_samples))
val_idx = lhc_grid[:n_val]
test_idx = lhc_grid[n_val : n_val + n_test]
train_idx = list(lhc_grid[n_val + n_test :]) + derivative_grid

split_dict = {
    'train': [int(idx) for idx in train_idx],
    'val': [int(idx) for idx in val_idx],
    'test': [int(idx) for idx in test_idx],
}
with open('../data/train_test_split.json', 'w') as f:
    json.dump(split_dict, f)
