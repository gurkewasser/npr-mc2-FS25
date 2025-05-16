import torch

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