from flwr.common import ndarrays_to_parameters
def powersgd_ndarrays_to_parameters(ndarrays):
    """Return the low_rank parameters from the provided list of ndarrays."""
    P = None
    Q = None

    parameters = ndarrays_to_parameters([P, Q])
    return parameters
