import numpy as np
from flwr.common import ndarrays_to_parameters

from utils.eval import calc_none_zero_data, calc_zero_data


def randomk_ndarrays_to_parameters(ndarrays, k=2):
    """Return the random-k parameters from the provided list of ndarrays."""
    randomk_ndarrays = []
    for ndarray in ndarrays:
        print("ndarray shape:", ndarray.shape)
        print("ndarray ndim:", ndarray.ndim)
        if ndarray.ndim <= 1 and k > ndarray.size:
            # Flatten the array and get the indices of the top-k elements
            indices = np.random.choice(ndarray.size, k, replace=False)
            # Create a new array with only the top-k elements
            rdk_array = np.zeros_like(ndarray, dtype=ndarray.dtype)
            np.put(rdk_array, indices, ndarray.flatten()[indices])
            randomk_ndarrays.append(rdk_array)
        else:
            # Get the shape of the ndarray
            shape = ndarray.shape
            rdk_array = np.zeros_like(ndarray, dtype=ndarray.dtype)
            # Iterate over the first n-2 dimensions
            for index in np.ndindex(shape[:-2]):
                # Get the sub-array corresponding to the last two dimensions
                sub_array = ndarray[index]
                # Flatten the array and get the indices of the top-k elements
                indices = np.random.choice(sub_array.size, k, replace=False)
                # Create a new array with only the top-k elements
                np.put(rdk_array[index], indices, sub_array.flatten()[indices])
            randomk_ndarrays.append(rdk_array)
        print("topk_array shape:", rdk_array.shape)
    bytes = calc_none_zero_data(randomk_ndarrays)
    print("none zero bytes:", bytes)
    zero_bytes = calc_zero_data(randomk_ndarrays)
    print("zero bytes:", zero_bytes)
    # return topk_ndarrays
    parameters = ndarrays_to_parameters(randomk_ndarrays)
    return parameters, bytes
