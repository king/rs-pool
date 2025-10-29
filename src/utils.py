# Copyright (C) king.com Ltd 2025
# License: Apache 2.0
import numpy as np
import random
import torch
import os

def set_seed(seed: int):
    # Set PYTHONHASHSEED environment variable for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Python built-in random module
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == "__main__":
    pass
