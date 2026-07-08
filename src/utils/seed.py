import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across the classic-ML and
    torch-based training pipelines."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
