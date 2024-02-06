import numpy as np

from ser.sparse import ndarrays_to_sparse_parameters, sparse_parameters_to_ndarrays


def topk_ndarrays_to_parameters(ndarrays, k=2):
    """Return the top-k parameters from the provided list of ndarrays."""
    topk_ndarrays = []
    for ndarray in ndarrays:
        print("ndarray shape:", ndarray.shape)
        if ndarray.ndim <= 2:
            # Flatten the array and get the indices of the top-k elements
            indices = np.argpartition(ndarray.flatten(), -k)[-k:]
            # Create a new array with only the top-k elements
            topk_array = np.zeros_like(ndarray, dtype=ndarray.dtype)
            np.put(topk_array, indices, ndarray.flatten()[indices])
            topk_ndarrays.append(topk_array)
        else:
            # Get the shape of the ndarray
            shape = ndarray.shape
            topk_array = np.zeros_like(ndarray, dtype=ndarray.dtype)
            # Iterate over the first n-2 dimensions
            for index in np.ndindex(shape[:-2]):
                # Get the sub-array corresponding to the last two dimensions
                sub_array = ndarray[index]
                # Flatten the array and get the indices of the top-k elements
                indices = np.argpartition(sub_array.flatten(), -k)[-k:]
                # Create a new array with only the top-k elements
                np.put(topk_array[index], indices, sub_array.flatten()[indices])
            topk_ndarrays.append(topk_array)
        print("topk_array shape:", topk_array.shape)

    parameters = ndarrays_to_sparse_parameters(topk_ndarrays)
    return parameters


def topk_parameters_to_ndarrays(parameters):
    """Return the list of ndarrays from the provided top-k parameters."""
    ndarrays = sparse_parameters_to_ndarrays(parameters)
    return ndarrays
