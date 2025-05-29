import torch
import numpy as np
import random
import os

def find_max_batch_size(model, texts, start=4, max_possible=1024):
    """
    Findet die maximale batch_size, die noch ohne Speicherfehler funktioniert.
    Nutzt binäre Suche (Binary Search).
    """
    low = start
    high = max_possible
    best = low

    while low <= high:
        mid = (low + high) // 2
        try:
            _ = model.encode(texts[:mid], batch_size=mid, show_progress_bar=False)
            best = mid
            low = mid + 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                high = mid - 1
            else:
                raise e
    return best


def set_seed(seed: int = 2277):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Or ':16:8' if out of memory

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False