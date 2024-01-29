import numpy as np

from .constants import DATA_DIR


def save_val(val: list[np.ndarray], path=DATA_DIR / "ret_val.npy"):
    np.save(path.as_posix(), np.array(val, dtype=object), allow_pickle=True)
    return
