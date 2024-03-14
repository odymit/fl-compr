import numpy as np


def calc_data(data: list[np.ndarray]):
    bytes = sum([p.nbytes for p in data])
    return bytes


def calc_none_zero_data(data: list[np.ndarray]):
    bytes = sum([p[abs(p) >= 1e-5].nbytes for p in data])
    return bytes


def calc_zero_data(data: list[np.ndarray]):
    bytes = sum([p[abs(p) < 1e-5].nbytes for p in data])
    return bytes
