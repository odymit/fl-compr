import numpy as np

from ser.sparse import ndarrays_to_sparse_parameters, sparse_parameters_to_ndarrays


def randomk_ndarrays_to_parameters(ndarrays, k=2):
    """Return the top-k parameters from the provided list of ndarrays."""
    randomk_ndarrays = []
    for ndarray in ndarrays:
        print("ndarray shape:", ndarray.shape)
        print("ndarray ndim:", ndarray.ndim)
        if ndarray.ndim <= 1:
            # Get the indices of the k random elements
            indices = np.random.choice(ndarray.size, k, replace=False)

            # Create a new ndarray of zeros with the same shape as the original
            randomk_array = np.zeros_like(ndarray)

            # Use np.unravel_index to convert flat indices to multi-dimensional indices
            multi_dim_indices = np.unravel_index(indices, ndarray.shape)

            # Set the k elements at the random indices to their original values
            randomk_array[multi_dim_indices] = ndarray[multi_dim_indices]
            randomk_ndarrays.append(randomk_array)
        else:
            # get the shape of the ndarray
            shape = ndarray.shape
            randomk_array = np.zeros_like(ndarray, dtype=ndarray.dtype)
            # iterate over the first n-2 dimensions
            for index in np.ndindex(shape[:-1]):
                # get the sub-array corresponding to the last two dimensions
                sub_array = ndarray[index]
                # get the indices of the k random elements
                indices = np.random.choice(sub_array.size, k, replace=False)
                # print("sub_array shape:", sub_array.shape)
                # print("indices:", indices)
                # use np.unravel_index to convert flat indices to multi-dimensional indices
                multi_dim_indices = np.unravel_index(indices, sub_array.shape)
                # set the k elements at the random indices to their original values
                randomk_array[index][multi_dim_indices] = sub_array[multi_dim_indices]
            randomk_ndarrays.append(randomk_array)
        print("randomk_array shape:", randomk_array.shape)
    # return randomk_ndarrays
    parameters = ndarrays_to_sparse_parameters(randomk_ndarrays)
    return parameters


def randomk_parameters_to_ndarrays(parameters):
    """Return the list of ndarrays from the provided top-k parameters."""
    ndarrays = sparse_parameters_to_ndarrays(parameters)
    return ndarrays
