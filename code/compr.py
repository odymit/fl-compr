import torch


def topk(k, model_state_dict):
    # create a new dict for the top-k parameters
    topk_state_dict = {}
    bytes = 0
    # get the top-k elements of each parameter
    for name, param in model_state_dict.items():
        # get real k for each matrix in the same settings of PowerSGD
        if param.ndim > 1:
            # get (m+n)*k if param have a shape of [m, n]
            real_k = sum(list(param.shape)) * k
        else:
            real_k = k
        bytes += real_k * param.element_size()
        # if k is larger than the number of elements in the parameter tensor, skip it
        if real_k > len(param.view(-1)):
            continue
        # flatten the parameter tensor and get the top-k elements
        abs_param = param.abs()
        _, topk_indices = torch.topk(abs_param.view(-1), real_k)

        # create a new tensor with the same shape as the parameter tensor, filled with zeros
        topk_param = torch.zeros_like(param)

        # replace the top-k positions in the new tensor with the top-k values
        topk_param.view(-1)[topk_indices] = param.view(-1)[topk_indices]

        # add the new parameter tensor to the new state dict
        topk_state_dict[name] = topk_param
    print("Bytes of top-k: ", bytes)
    return topk_state_dict


def randomk(k, model_state_dict):
    # create a new dict for the random-k parameters
    randomk_state_dict = {}
    bytes = 0
    # get the random-k elements of each parameter
    for name, param in model_state_dict.items():
        # get real k for each matrix in the same settings of PowerSGD
        if param.ndim > 1:
            # get (m+n)*k if param have a shape of [m, n]
            real_k = sum(list(param.shape)) * k
        else:
            real_k = k
        bytes += real_k * param.element_size()
        # if k is larger than the number of elements in the parameter tensor, skip it
        if real_k > len(param.view(-1)):
            continue
        # flatten the parameter tensor and get the random-k elements
        _, randomk_indices = torch.sort(torch.randperm(len(param.view(-1)))[:real_k])

        # create a new tensor with the same shape as the parameter tensor, filled with zeros
        randomk_param = torch.zeros_like(param)

        # replace the random-k positions in the new tensor with the random-k values
        randomk_param.view(-1)[randomk_indices] = param.view(-1)[randomk_indices]

        # add the new parameter tensor to the new state dict
        randomk_state_dict[name] = randomk_param
    print("Bytes of random-k: ", bytes)
    return randomk_state_dict
