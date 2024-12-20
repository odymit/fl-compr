import json
import os
import random
from math import prod
from time import time

import numpy as np
import resnet_model
import torch
from tqdm import tqdm


def powersgd_decompr(p, q):
    ret = torch.mm(p, q.t())
    # print("decompr powersgd: ", ret)
    return ret


def orthogonalize(matrix: torch.Tensor, eps=torch.tensor(1e-16)):
    if matrix.shape[-1] == 1:
        matrix.div_(torch.maximum(matrix.norm(), eps))
    else:
        matrix.copy_(torch.linalg.qr(matrix).Q)
    return matrix


def init_Q(matrix, rank):
    Q_shape = (matrix.shape[1], rank)
    Q = torch.randn(Q_shape)
    return Q


def powersgd_compr(rank, matrix, epsilon=0.01):
    """do the compression and return the P and Q"""
    error = np.Inf
    Q = init_Q(matrix, rank)
    while error >= epsilon:
        # assume param in shape [n, m], then Q in shape [m, r]
        Q = Q.to(matrix.device)
        # print("Q: ", Q)
        P = torch.mm(matrix, Q)
        # print("P: ", P)
        P_hat = orthogonalize(P)
        # print("P_hat: ", P_hat)
        Q_new = torch.mm(matrix.t(), P_hat)
        # print("Q_new: ", Q_new)
        error = torch.norm(Q - Q_new)
        # print("error: ", error)
        Q = Q_new
    return P_hat, Q


def crack(integer):
    start = int(np.sqrt(integer))
    factor = integer / start

    def is_integer(x):
        if x == int(x):
            return True
        else:
            return False

    while not is_integer(factor):
        start += 1
        factor = integer / start
    return start, int(factor)


def flpsgd(global_model, recieved_model, conf, e, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bring out the first K gradient
    active_recieved = recieved_model[: conf["k"]]
    # average without weight
    global_gradient = global_model.state_dict()
    for name, data in global_gradient.items():
        global_gradient[name] = torch.zeros_like(data).to(device).float()
    if conf["model_name"] == "resnet18":
        gra = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "vgg16":
        gra = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "CNN":
        gra = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "LSTM":
        gra = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
    else:
        pass

    k = args.k
    print(active_recieved)
    org_bytes = 0
    bytes = 0
    error = 0
    time_cost = 0
    for name, data in global_model.state_dict().items():
        for gra_way in active_recieved:
            gra.load_state_dict(torch.load(gra_way[1]))
            gra_state = gra.state_dict()
            param = gra_state[name]
            org_bytes += param.numel() * param.element_size()
            start = time()

            # do the compression
            if args.gradient:
                info = gra_state[name] - global_gradient[name]
            else:
                info = gra_state[name]

            if info.ndim <= 1:
                # if ndim <= 1, take it uncompressed
                update_layer = info / conf["k"]
                global_gradient[name] += update_layer
                continue

            # get P & Q
            # print("gradient shape: ", info.shape)
            if args.square:
                row, col = crack(prod(list(info.shape)))
            else:
                row, col = info.shape[0], -1
            matrix = info.view(row, col)

            # get the error feedback
            if args.error_feedback:
                global efm
                print(efm.keys())
                # name in efm, it has previous error feedback value
                if name in efm:
                    matrix += efm[name]
                # name not in efm, have not initialize the efm of name
                else:
                    efm[name] = torch.zeros_like(matrix)

            P, Q = powersgd_compr(k, matrix)
            # counting time
            end = time()
            duration = end - start
            time_cost += duration
            # counting bytes
            bytes += P.numel() * P.element_size() + Q.numel() * Q.element_size()
            matrix_decompr = powersgd_decompr(P, Q)
            # counting error
            error += torch.norm(matrix - matrix_decompr).pow(2)

            # update error feedback
            if args.error_feedback:
                print("updating efm")
                efm[name] = matrix - matrix_decompr
            # update the global gradient
            update_layer = matrix_decompr.view(info.shape) / conf["k"]
            global_gradient[name] += update_layer
        if data.type() != global_gradient[name].type():
            global_gradient[name] = torch.round(global_gradient[name]).to(torch.int64)
        else:
            pass
        data.copy_(global_gradient[name])

    # save the logs
    filename = "powersgd_results.json"
    if not os.path.exists(filename):
        with open("powersgd_results.json", "w") as f:
            logs = {}
            json.dump(logs, f)
    print("path of powersgd results: ", os.path.abspath(filename))
    # load old logs
    old_logs = {}
    with open(filename, "r") as f:
        logs = {}
        logs[args.global_epoch] = {}
        logs[args.global_epoch]["org_bytes"] = org_bytes
        logs[args.global_epoch]["bytes"] = bytes
        logs[args.global_epoch]["error"] = error.sqrt().item()
        logs[args.global_epoch]["time_cost"] = time_cost

        old_logs = json.load(f)
        old_logs.update(logs)
    # append and save new logs
    with open(filename, "w") as f:
        json.dump(old_logs, f)
    print("Bytes before compression of PowerSGD: ", org_bytes)
    print("Bytes after compression of PowerSGD: ", bytes)
    print("Error of PowerSGD: ", error.sqrt().item())
    print("Time cost of compression: ", time_cost)
    return global_model


def fltopk(global_model, recieved_model, conf, e, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bring out the first K gradient
    active_recieved = recieved_model[: conf["k"]]
    # average without weight
    global_gradient = global_model.state_dict()
    for name, data in global_gradient.items():
        global_gradient[name] = data.to(device).float()
    if conf["model_name"] == "resnet18":
        gra = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "vgg16":
        gra = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "CNN":
        gra = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "LSTM":
        gra = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
    else:
        pass

    print(active_recieved)
    k = args.k
    org_bytes = 0
    bytes = 0
    error = 0
    time_cost = 0
    for name, data in tqdm(global_model.state_dict().items(), desc="aggregating"):
        for gra_way in active_recieved:
            gra.load_state_dict(torch.load(gra_way[1]))
            gra_state = gra.state_dict()
            param = gra_state[name]
            org_bytes += param.numel() * param.element_size()
            start = time()

            gradient = gra_state[name] - global_gradient[name]
            if param.ndim > 1:
                # get (m+n)*k if param have a shape of [m, n]
                real_k = min(sum(list(param.shape)) * k, len(param.view(-1)))
            else:
                # if ndim == 1, take it uncompressed
                update_layer = gradient / conf["k"]
                global_gradient[name] += update_layer
                continue

            # get random k indices
            abs_param = gradient.abs()
            _, topk_indices = torch.topk(abs_param.view(-1), real_k)
            # get mask
            mask = torch.zeros_like(gradient)
            flat_mask = mask.view(-1)
            flat_mask[topk_indices] = 1
            mask = mask.bool()
            # counting time
            end = time()
            duration = end - start
            time_cost += duration
            # counting bytes
            bytes += mask.sum().item() * param.element_size()
            # counting error
            matrix = torch.zeros_like(param)
            matrix[mask] = param[mask]
            error += torch.norm(param - matrix).pow(2).item()
            # g1 = 1/2(c1-g0)*mask1 + 1/2(c2-g0)*mask2 + g0
            update_layer = gradient / conf["k"]
            global_gradient[name][mask] += update_layer[mask]

        if data.type() != global_gradient[name].type():
            global_gradient[name] = torch.round(global_gradient[name]).to(torch.int64)
        else:
            pass
        data.copy_(global_gradient[name])

    # save the logs
    filename = "topk_results.json"
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            logs = {}
            json.dump(logs, f)
    print("path of topk results: ", os.path.abspath(filename))
    old_logs = {}
    error = torch.tensor(error)
    with open(filename, "r") as f:
        logs = {}
        logs[args.global_epoch] = {}
        logs[args.global_epoch]["org_bytes"] = org_bytes
        logs[args.global_epoch]["bytes"] = bytes
        logs[args.global_epoch]["error"] = error.sqrt().item()
        logs[args.global_epoch]["time_cost"] = time_cost

        old_logs = json.load(f)
        old_logs.update(logs)
    with open(filename, "w") as f:
        json.dump(old_logs, f)
    print("Bytes before compression of TopK: ", org_bytes)
    print("Bytes after compression of TopK: ", bytes)
    print("Error of TopK: ", error.sqrt().item())
    print("Time cost of compression: ", time_cost)
    return global_model


def flrandomk(global_model, recieved_model, conf, e, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bring out the first K gradient
    active_recieved = recieved_model[: conf["k"]]
    # average without weight
    global_gradient = global_model.state_dict()
    for name, data in global_gradient.items():
        global_gradient[name] = data.to(device).float()
    if conf["model_name"] == "resnet18":
        gra = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "vgg16":
        gra = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "CNN":
        gra = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "LSTM":
        gra = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
    else:
        pass

    print(active_recieved)
    k = args.k
    org_bytes = 0
    bytes = 0
    error = 0
    time_cost = 0
    for name, data in tqdm(global_model.state_dict().items(), desc="aggregating"):
        for gra_way in active_recieved:
            gra.load_state_dict(torch.load(gra_way[1]))
            gra_state = gra.state_dict()
            param = gra_state[name]
            org_bytes += param.numel() * param.element_size()
            start = time()

            gradient = gra_state[name] - global_gradient[name]
            if param.ndim > 1:
                # get (m+n)*k if param have a shape of [m, n]
                real_k = sum(list(param.shape)) * k
            else:
                # if ndim == 1, take it uncompressed
                update_layer = gradient / conf["k"]
                global_gradient[name] += update_layer
                continue

            # get random k indices
            flat_param = gradient.view(-1)
            num_elements = flat_param.numel()
            perm = torch.randperm(num_elements)
            randomk_indices = perm[:real_k]
            # get mask
            mask = torch.zeros_like(gradient)
            flat_mask = mask.view(-1)
            flat_mask[randomk_indices] = 1
            mask = mask.bool()
            # counting time
            end = time()
            duration = end - start
            time_cost += duration
            # counting bytes
            bytes += mask.sum().item() * param.element_size()
            # counting error
            matrix = torch.zeros_like(param)
            matrix[mask] = param[mask]
            error += torch.norm(param - matrix).pow(2).item()
            # g1 = 1/2(c1-g0)*mask1 + 1/2(c2-g0)*mask2 + g0
            update_layer = gradient / conf["k"]
            global_gradient[name][mask] += update_layer[mask]

        if data.type() != global_gradient[name].type():
            global_gradient[name] = torch.round(global_gradient[name]).to(torch.int64)
        else:
            pass
        data.copy_(global_gradient[name])

    # save the logs
    filename = "randomk_results.json"
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            logs = {}
            json.dump(logs, f)
    print("path of randomk results: ", os.path.abspath(filename))
    old_logs = {}
    error = torch.tensor(error)
    with open(filename, "r") as f:
        logs = {}
        logs[args.global_epoch] = {}
        logs[args.global_epoch]["org_bytes"] = org_bytes
        logs[args.global_epoch]["bytes"] = bytes
        logs[args.global_epoch]["error"] = error.sqrt().item()
        logs[args.global_epoch]["time_cost"] = time_cost

        old_logs = json.load(f)
        old_logs.update(logs)
    with open(filename, "w") as f:
        json.dump(old_logs, f)
    print("Bytes before compression of RandomK: ", org_bytes)
    print("Bytes after compression of RandomK: ", bytes)
    print("Error of RandomK: ", error.sqrt().item())
    print("Time cost of compression: ", time_cost)
    return global_model


def flrandomblock(global_model, recieved_model, conf, e, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bring out the first K gradient
    active_recieved = recieved_model[: conf["k"]]
    # average without weight
    global_gradient = global_model.state_dict()
    for name, data in global_gradient.items():
        global_gradient[name] = data.to(device).float()
    if conf["model_name"] == "resnet18":
        gra = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "vgg16":
        gra = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "CNN":
        gra = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "LSTM":
        gra = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
    else:
        pass

    print(active_recieved)
    k = args.k
    org_bytes = 0
    bytes = 0
    error = 0
    time_cost = 0
    for name, data in tqdm(global_model.state_dict().items(), desc="aggregating"):
        for gra_way in active_recieved:
            gra.load_state_dict(torch.load(gra_way[1]))
            gra_state = gra.state_dict()
            param = gra_state[name]
            org_bytes += param.numel() * param.element_size()
            start = time()

            gradient = gra_state[name] - global_gradient[name]
            if param.ndim > 1:
                # get (m+n)*k if param have a shape of [m, n]
                real_k = sum(list(param.shape)) * k
            else:
                # if ndim == 1, take it uncompressed
                update_layer = gradient / conf["k"]
                global_gradient[name] += update_layer
                continue

            # get random k indices
            flat_param = gradient.view(-1)
            num_elements = flat_param.numel()
            # perm = torch.randperm(num_elements)
            s = random.randint(0, num_elements - real_k - 1)
            randomk_indices = torch.tensor([s + i for i in range(real_k)])
            # get mask
            mask = torch.zeros_like(gradient)
            flat_mask = mask.view(-1)
            flat_mask[randomk_indices] = 1
            mask = mask.bool()
            # counting time
            end = time()
            duration = end - start
            time_cost += duration
            # counting bytes
            bytes += mask.sum().item() * param.element_size()
            # counting error
            matrix = torch.zeros_like(param)
            matrix[mask] = param[mask]
            error += torch.norm(param - matrix).pow(2).item()
            # g1 = 1/2(c1-g0)*mask1 + 1/2(c2-g0)*mask2 + g0
            update_layer = gradient / conf["k"]
            global_gradient[name][mask] += update_layer[mask]

        if data.type() != global_gradient[name].type():
            global_gradient[name] = torch.round(global_gradient[name]).to(torch.int64)
        else:
            pass
        data.copy_(global_gradient[name])

    # save the logs
    filename = "randomblock_results.json"
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            logs = {}
            json.dump(logs, f)
    print("path of randomk block results: ", os.path.abspath(filename))
    old_logs = {}
    error = torch.tensor(error)
    with open(filename, "r") as f:
        logs = {}
        logs[args.global_epoch] = {}
        logs[args.global_epoch]["org_bytes"] = org_bytes
        logs[args.global_epoch]["bytes"] = bytes
        logs[args.global_epoch]["error"] = error.sqrt().item()
        logs[args.global_epoch]["time_cost"] = time_cost

        old_logs = json.load(f)
        old_logs.update(logs)
    with open(filename, "w") as f:
        json.dump(old_logs, f)
    print("Bytes before compression of RandomK: ", org_bytes)
    print("Bytes after compression of RandomK: ", bytes)
    print("Error of RandomK: ", error.sqrt().item())
    print("Time cost of compression: ", time_cost)
    return global_model


def flavg(global_model, recieved_model, conf, e):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bring out the first K gradient
    active_recieved = recieved_model[: conf["k"]]
    # average without weight
    global_gradient = global_model.state_dict()
    for name, data in global_gradient.items():
        global_gradient[name] = torch.zeros_like(data).to(device).float()
    if conf["model_name"] == "resnet18":
        gra = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "vgg16":
        gra = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "CNN":
        gra = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "LSTM":
        gra = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
    else:
        pass

    print(active_recieved)
    for name, data in global_model.state_dict().items():
        for gra_way in active_recieved:
            gra.load_state_dict(torch.load(gra_way[1]))
            gra_state = gra.state_dict()
            update_layer = gra_state[name] / conf["k"]
            global_gradient[name] += update_layer

        if data.type() != global_gradient[name].type():
            global_gradient[name] = torch.round(global_gradient[name]).to(torch.int64)
        else:
            pass
        data.copy_(global_gradient[name])

    return global_model


def flavg_hierarchy_aggr(global_model, recieved_model, conf, e, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("args:", args)
    # bring out the first K gradient
    active_recieved = recieved_model[: conf["k"]]
    # average without weight
    global_gradient = global_model.state_dict()
    for name, data in global_gradient.items():
        global_gradient[name] = torch.zeros_like(data).to(device).float()
    if conf["model_name"] == "resnet18":
        gra = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "vgg16":
        gra = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "CNN":
        gra = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "LSTM":
        gra = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
    else:
        pass

    print("active_recieved: ", active_recieved)
    active_worker_idx_lst = []
    for idx, path in active_recieved:
        active_worker_idx_lst.append(idx)
    # if there are 20 clients, and 10 are at least selected
    # split group
    # assume groups k = 4
    k_groups = 4
    num_workers_in_each_grpus = len(args.worker_conf) // k_groups
    group_to_workers = [[] * k_groups]
    group_idx = 1
    while group_idx <= k_groups:
        start = (group_idx - 1) * num_workers_in_each_grpus
        end = group_idx * num_workers_in_each_grpus
        if group_idx == k_groups:
            end = len(args.worker_conf)
        for worker_idx in range(start, end):
            if worker_idx in active_worker_idx_lst:
                group_to_workers[group_idx - 1].append(worker_idx)
        group_idx += 1

    print("splited workers: ", group_to_workers)

    def get_gra_path(active_recieved, worker_idx):
        for idx, path in active_recieved:
            if idx == worker_idx:
                return path
        else:
            raise FileNotFoundError(
                "worker_idx: {idx} not found in {path}.".format(
                    idx=worker_idx, path=str(active_recieved)
                )
            )

    for name, data in global_model.state_dict().items():
        ds_idx_to_gra_state_dict = {}
        # aggregate edge group
        for group_idx in range(k_groups):
            worker_idx_lst_in_cur_group = group_to_workers[group_idx]
            length = len(worker_idx_lst_in_cur_group)
            if length == 0:
                ds_idx_to_gra_state_dict.update({group_idx: None})
            elif length == 1:
                gra_path = get_gra_path(active_recieved, worker_idx_lst_in_cur_group[0])
                gra.load_state_dict(torch.load(gra_path))
                gra_state = gra.state_dict()
                ds_idx_to_gra_state_dict.update({group_idx: gra_state[name]})
            else:
                edge_aggr_layer = None
                for i in worker_idx_lst_in_cur_group:
                    gra_path = get_gra_path(active_recieved, i)
                    gra.load_state_dict(torch.load(gra_path))
                    gra_state = gra.state_dict()
                    if not edge_aggr_layer:
                        edge_aggr_layer = torch.zeros_like(gra_state[name])
                    update_layer = gra_state[name] / length
                    edge_aggr_layer += update_layer
                ds_idx_to_gra_state_dict.update({group_idx: edge_aggr_layer})
        # aggregate central
        for update_layer in ds_idx_to_gra_state_dict.values():
            global_gradient[name] += update_layer

        if data.type() != global_gradient[name].type():
            global_gradient[name] = torch.round(global_gradient[name]).to(torch.int64)
        else:
            pass
        data.copy_(global_gradient[name])

    return global_model
