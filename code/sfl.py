import argparse
import csv
import json
import random
import time
from pathlib import Path

import resnet_model
import torch
from aggre import flavg, flpsgd, flrandomblock, flrandomk, fltopk
from data_pre import get_dataset


# evaluate the model by data from data_loader
def eval_model(model, data_loader):
    print("Start evaluating the model!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0

    for batch_id, batch in enumerate(data_loader):
        data, target = batch
        dataset_size += data.size()[0]
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
        output = model(data)

        total_loss += torch.nn.functional.cross_entropy(
            output, target, reduction="sum"
        ).item()

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))
    total_l = total_loss / dataset_size

    return acc, total_l


# aggregrate the model
def aggregate_model(global_model, recieved_model, conf, e, args):
    if args.aggregate == "flavg":
        print("Using default aggregate mode(fl avg).")
        return flavg(global_model, recieved_model, conf, e)
    if args.aggregate == "randomk":
        print("Using compr aggregate mode(fl randomk).")
        return flrandomk(global_model, recieved_model, conf, e, args)
    elif args.aggregate == "randomblock":
        print("Using compr aggregate mode(fl randomblock).")
        return flrandomblock(global_model, recieved_model, conf, e, args)
    elif args.aggregate == "topk":
        print("Using compr aggregate mode(fl topk).")
        return fltopk(global_model, recieved_model, conf, e, args)
    elif args.aggregate == "powersgd":
        print("Using PowerSGD compr mode.")
        return flpsgd(global_model, recieved_model, conf, e, args)
    else:
        print("Error: aggregate mode not found! Using default aggregate mode.")
        return flavg(global_model, recieved_model, conf, e)


# train model
def train_model(model, optimizer, data_loader, conf, seq, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    gra_dict = {}
    for name, data in model.state_dict().items():
        gra_dict[name] = model.state_dict()[name].clone()

    for e in range(conf["local_epochs"]):
        for batch_id, batch in enumerate(data_loader):
            data, target = batch
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=conf["clip"])
            optimizer.step()

            if batch_id % 10 == 0:
                print("\t \t Finish ", batch_id, "/", len(data_loader), "batches.")

        print("\t Client", seq, " finsh ", e, " epoches train! ")

    # communication with server
    path = Path(__file__).parent / "sfl" / args.aggregate
    filename = "gradient_" + str(seq) + ".pt"
    if not path.exists():
        path.mkdir()
    state_dict = model.state_dict()
    # filename = str(args.global_epoch) + "_" + filename
    torch.save(state_dict, path / filename)

    return model


# main function
def main():
    # get config
    parser = argparse.ArgumentParser(description="Federated Learning")
    parser.add_argument("-c", "--conf", dest="conf")
    parser.add_argument(
        "-k",
        "--k",
        dest="k",
        type=int,
        default=2,
        help="the k value of top-k or random-k, equal to r in PowerSGD",
    )
    parser.add_argument(
        "-a",
        "--aggregate",
        dest="aggregate",
        default="flavg",
        choices=["flavg", "randomk", "randomblock", "topk", "powersgd"],
        help="the aggregate mode, default is flavg",
    )
    parser.add_argument(
        "--gradient",
        default=False,
        action="store_true",
        help="just for the weight communication method",
    )
    parser.add_argument(
        "--square",
        default=False,
        action="store_true",
        help="just for the flpsgd method",
    )
    parser.add_argument(
        "--error_feedback",
        dest="error_feedback",
        default=False,
        action="store_true",
        help="trigger the error feedback",
    )
    args = parser.parse_args()
    with open(args.conf, "r", encoding="utf-8") as f:
        conf = json.load(f)

    # get evaluation dataloader for server
    eval_loader, eval_loader_list = get_dataset(
        "../dataset/", conf["type"], "s", conf, -1
    )

    # set workers
    workers = conf["no_models"]  # amount
    # each worker's config:
    # {number(int) : [
    #   resource(int),
    #   data_lodaer(torch.loader),
    #   time_stamp(int),
    #   global_stamp(int),
    #   newest model(str),
    #   psgd_q_prev([tensor])
    # ],
    # }
    worker_conf = {}

    for i in range(workers):
        resource = 1
        print("Client ", i, " has ", resource, " resource.")
        time.sleep(0.5)
        worker_conf[i] = [
            resource,
            get_dataset("../dataset/", conf["type"], "c", conf, i),
            0,
            0,
            "./sfl/global_model_0.pt",
            None,
        ]
    args.worker_conf = worker_conf

    # workflow
    global_epoch = 0
    have_recieved_model = []
    time_clock = 0
    uploaded_model = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize the model
    if conf["model_name"] == "resnet18":
        global_model = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "vgg16":
        global_model = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "CNN":
        global_model = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
    elif conf["model_name"] == "LSTM":
        global_model = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
    else:
        pass
    torch.save(global_model.state_dict(), "./sfl/global_model_0.pt")

    # start training
    start_time = time.time()
    while global_epoch < conf["global_epochs"]:

        print("\nGlobal Epoch ", global_epoch, " Starts! \n")

        active_client = random.sample(range(conf["no_models"]), conf["k"])

        for client_seq_number in range(workers):

            if client_seq_number in active_client:
                # start train
                print("\t Client ", client_seq_number, "start train!")
                # load newest global model and dataloader
                train_loader = worker_conf[client_seq_number][1]
                using_train_model = worker_conf[client_seq_number][4]
                if conf["model_name"] == "resnet18":
                    local_model = resnet_model.ResNet18(num=conf["CLASS_NUM"]).to(
                        device
                    )
                elif conf["model_name"] == "vgg16":
                    local_model = resnet_model.Vgg16(num=conf["CLASS_NUM"]).to(device)
                elif conf["model_name"] == "CNN":
                    local_model = resnet_model.CNN(num=conf["CLASS_NUM"]).to(device)
                elif conf["model_name"] == "LSTM":
                    local_model = resnet_model.LSTM(num=conf["CLASS_NUM"]).to(device)
                else:
                    pass
                local_model.load_state_dict(torch.load(using_train_model))

                # train
                optimizer = torch.optim.SGD(
                    local_model.parameters(),
                    lr=conf["local_lr"],
                    momentum=conf["local_momentum"],
                )
                args.global_epoch = global_epoch
                local_model = train_model(
                    local_model,
                    optimizer,
                    train_loader,
                    conf,
                    client_seq_number,
                    args,
                )
                # compute the updation
                print("Client ", client_seq_number, "finish train and upload gradient!")
                gra = (
                    "./sfl/"
                    + args.aggregate
                    + "/gradient_"
                    + str(client_seq_number)
                    + ".pt"
                )
                have_recieved_model.append(
                    [client_seq_number, gra]
                )  # update the model to server
                uploaded_model += 1

            else:
                print(
                    "Client ", client_seq_number, "keep training!"
                )  # keep training(idling)

        recieved_amount = len(have_recieved_model)
        print(
            "\nUsing ",
            time_clock,
            " time clocks and recieve ",
            recieved_amount,
            " models! \n",
        )

        time.sleep(0.5)

        if recieved_amount < conf["k"]:
            print(
                "Waiting for enough models! Need ",
                conf["k"],
                ", but recieved ",
                recieved_amount,
            )  # have not recieved enough models, keep waiting
        else:
            print(
                "Having recieved enough models. Need ",
                conf["k"],
                ", and recieved ",
                recieved_amount,
            )
            # aggregrate
            global_model = aggregate_model(
                global_model, have_recieved_model, conf, global_epoch, args
            )

            with open(
                "sfl_"
                + str(args.k)
                + "_"
                + "aggre="
                + args.aggregate
                + "_"
                + conf["model_name"]
                + "_"
                + conf["type"]
                + "_size_with"
                + "_alpha_"
                + str(conf["alpha"])
                + "_clip_"
                + str(conf["clip"])
                + ".csv",
                mode="a+",
                newline="",
            ) as file:
                writer = csv.writer(file)
                writer.writerow([global_epoch, uploaded_model])

            # evaluation
            total_acc, total_loss = eval_model(global_model, eval_loader)
            this_time = time.time()
            print(
                "Global Epoch ",
                global_epoch,
                "\t total loss: ",
                total_loss,
                " \t total acc: ",
                total_acc,
            )
            with open(
                "sfl_"
                + str(args.k)
                + "_"
                + "aggre="
                + args.aggregate
                + "_"
                + conf["model_name"]
                + "_"
                + conf["type"]
                + "_acc_with"
                + "_alpha_"
                + str(conf["alpha"])
                + ".csv",
                mode="a+",
                newline="",
            ) as file:
                writer = csv.writer(file)
                # for row in ret:
                writer.writerow(
                    [global_epoch, total_acc, total_loss, this_time - start_time]
                )

            # save global model and add the epoch
            have_recieved_model = have_recieved_model[conf["k"] :]
            global_epoch += 1
            torch.save(
                global_model.state_dict(),
                "./sfl/global_model_" + str(global_epoch) + ".pt",
            )

            # notice the newest global model to each client
            for client_seq_number in range(workers):
                worker_conf[client_seq_number][4] = (
                    "./sfl/global_model_" + str(global_epoch) + ".pt"
                )

            print("Finish aggregrate and leave ", len(have_recieved_model), " models!")

        time.sleep(0.5)


if __name__ == "__main__":
    sfl_path = Path(__file__).parent / "sfl"
    if not sfl_path.exists():
        sfl_path.mkdir()
    main()
