import json
import os
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
            matrix = info.view(info.shape[0], -1)
            P, Q = powersgd_compr(k, matrix)
            # counting time
            end = time()
            duration = end - start
            time_cost += duration
            # counting bytes
            bytes += P.numel() * P.element_size() + Q.numel() * Q.element_size()
            # counting error
            error += torch.norm(matrix - powersgd_decompr(P, Q)).pow(2)

            # update the global gradient
            update_layer = powersgd_decompr(P, Q).view(info.shape) / conf["k"]
            global_gradient[name] += update_layer
        if data.type() != global_gradient[name].type():
            global_gradient[name] = torch.round(global_gradient[name]).to(torch.int64)
        else:
            pass
        data.copy_(global_gradient[name])

    # save the logs
    if not os.path.exists("powersgd.log"):
        with open("powersgd.log", "w") as f:
            logs = {}
            json.dump(logs, f)
    print("path of powersgd.log: ", os.path.abspath("powersgd.log"))
    old_logs = {}
    with open("powersgd.log", "r") as f:
        logs = {}
        logs[args.global_epoch] = {}
        logs[args.global_epoch]["org_bytes"] = org_bytes
        logs[args.global_epoch]["bytes"] = bytes
        logs[args.global_epoch]["error"] = error.sqrt().item()
        logs[args.global_epoch]["time_cost"] = time_cost

        old_logs = json.load(f)
        old_logs.update(logs)
    with open("powersgd.log", "w") as f:
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
            error += torch.norm(param - param[mask]).pow(2)
            # g1 = 1/2(c1-g0)*mask1 + 1/2(c2-g0)*mask2 + g0
            update_layer = gradient / conf["k"]
            global_gradient[name][mask] += update_layer[mask]

        if data.type() != global_gradient[name].type():
            global_gradient[name] = torch.round(global_gradient[name]).to(torch.int64)
        else:
            pass
        data.copy_(global_gradient[name])

    # save the logs
    if not os.path.exists("topk.log"):
        with open("topk.log", "w") as f:
            logs = {}
            json.dump(logs, f)
    print("path of topk.log: ", os.path.abspath("topk.log"))
    old_logs = {}
    with open("topk.log", "r") as f:
        logs = {}
        logs[args.global_epoch] = {}
        logs[args.global_epoch]["org_bytes"] = org_bytes
        logs[args.global_epoch]["bytes"] = bytes
        logs[args.global_epoch]["error"] = error.sqrt().item()
        logs[args.global_epoch]["time_cost"] = time_cost

        old_logs = json.load(f)
        old_logs.update(logs)
    with open("topk.log", "w") as f:
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
            error += torch.norm(param - param[mask]).pow(2)
            # g1 = 1/2(c1-g0)*mask1 + 1/2(c2-g0)*mask2 + g0
            update_layer = gradient / conf["k"]
            global_gradient[name][mask] += update_layer[mask]

        if data.type() != global_gradient[name].type():
            global_gradient[name] = torch.round(global_gradient[name]).to(torch.int64)
        else:
            pass
        data.copy_(global_gradient[name])

    # save the logs
    if not os.path.exists("randomk.log"):
        with open("randomk.log", "w") as f:
            logs = {}
            json.dump(logs, f)
    print("path of randomk.log: ", os.path.abspath("randomk.log"))
    old_logs = {}
    with open("randomk.log", "r") as f:
        logs = {}
        logs[args.global_epoch] = {}
        logs[args.global_epoch]["org_bytes"] = org_bytes
        logs[args.global_epoch]["bytes"] = bytes
        logs[args.global_epoch]["error"] = error.sqrt().item()
        logs[args.global_epoch]["time_cost"] = time_cost

        old_logs = json.load(f)
        old_logs.update(logs)
    with open("randomk.log", "w") as f:
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
