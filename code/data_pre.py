# -*- coding: utf-8 -*-

from torchvision import datasets, transforms
import numpy as np
import torch
import random
import json

# 获取数据集
def get_dataset(dir, name, roll, conf, user_id):

    if torch.cuda.is_available():
        pin = True
    else:
        pin = False

    if name == 'Shakespeare':
        class Mydataset(torch.utils.data.Dataset):

            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.idx = list()
                for item in x:
                    self.idx.append(item)
                pass

            def __getitem__(self, index):
                input_data = self.idx[index]
                target = self.y[index]
                return input_data, target

            def __len__(self):
                return len(self.idx)

        ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
        NUM_LETTERS = len(ALL_LETTERS)

        def word_to_indices(word):
            indices = []
            for c in word:
                indices.append(ALL_LETTERS.find(c))
            return indices


        def letter_to_vec(letter):
            index = ALL_LETTERS.find(letter)
            return index

        with open("./all_data.json", "r+") as f:
            data_conf = json.load(f)
        char = data_conf["users"]
        plays = data_conf["hierarchies"]


        train_data_x = []
        train_data_y = []
        test_data_x  = []
        test_data_y  = []
        if conf["non_iid"] == "iid":
            if roll == "s":
                for i in range(200):
                    uid = np.random.randint(0,1129)
                    usr = char[uid]
                    x = data_conf["user_data"][usr]["x"]
                    y = data_conf["user_data"][usr]["y"]
                    data_x = [word_to_indices(word) for word in x]
                    data_y = [letter_to_vec(c) for c in y]
                    # data_x = torch.LongTensor(data_x)
                    # data_y = torch.LongTensor(data_y)

                    test_partition = int(len(data_x) * 0.8)

                    test_data_x += data_x[test_partition:]
                    test_data_y += data_y[test_partition:]

                test_data_x = torch.LongTensor(test_data_x)
                test_data_y = torch.LongTensor(test_data_y)

                test_dataset = Mydataset(test_data_x, test_data_y) 

                return torch.utils.data.DataLoader(test_dataset,batch_size=conf["batch_size"],shuffle=True, pin_memory = pin), test_dataset
            else:
                print("Process the data of ", user_id)
                uid_list = set()
                uid1 = np.random.randint(0,1129)
                uid_list.add(uid1)
                usr1 = char[uid1]
                x = data_conf["user_data"][usr1]["x"] 
                y = data_conf["user_data"][usr1]["y"] 
                for i in range(10):
                    uid = np.random.randint(0,1129)
                    while uid in uid_list:
                        uid = np.random.randint(0,1129)
                    usr = char[uid]
                    x += data_conf["user_data"][usr]["x"] 
                    y += data_conf["user_data"][usr]["y"] 

                data_x = [word_to_indices(word) for word in x]
                data_y = [letter_to_vec(c) for c in y]
                    # data_x = torch.LongTensor(data_x)
                    # data_y = torch.LongTensor(data_y)

                train_partition = int(len(data_x) * 0.8)

                train_data_x += data_x[:train_partition]
                train_data_y += data_y[:train_partition]

                train_data_x = torch.LongTensor(train_data_x)
                train_data_y = torch.LongTensor(train_data_y)

                train_dataset = Mydataset(train_data_x, train_data_y)
                print(user_id, "has data of ", len(train_data_x))
                return torch.utils.data.DataLoader(train_dataset, batch_size=conf["batch_size"], pin_memory = pin,drop_last=True)
        else:
            if roll == "s":
                sample_user = torch.load("Shakespear_train_600_frac_08.pt")
                rest_user = torch.load("Shakespear_test_600_frac_08.pt")
                for i in range(len(sample_user)):
                    usr = char[sample_user[i]]
                    x = data_conf["user_data"][usr]["x"]
                    y = data_conf["user_data"][usr]["y"]
                    data_x = [word_to_indices(word) for word in x]
                    data_y = [letter_to_vec(c) for c in y]
                    # data_x = torch.LongTensor(data_x)
                    # data_y = torch.LongTensor(data_y)

                    test_partition = int(len(data_x) * 0.8)

                    test_data_x += data_x[test_partition:]
                    test_data_y += data_y[test_partition:]

                for i in range(len(rest_user)):
                    usr = char[rest_user[i]]
                    x = data_conf["user_data"][usr]["x"]
                    y = data_conf["user_data"][usr]["y"]
                    data_x = [word_to_indices(word) for word in x]
                    data_y = [letter_to_vec(c) for c in y]
                    # data_x = torch.LongTensor(data_x)
                    # data_y = torch.LongTensor(data_y)

                    test_partition = int(len(data_x) * 0.8)

                    test_data_x += data_x[test_partition:]
                    test_data_y += data_y[test_partition:]

                test_data_x = torch.LongTensor(test_data_x)
                test_data_y = torch.LongTensor(test_data_y)

                test_dataset = Mydataset(test_data_x, test_data_y) 

                return torch.utils.data.DataLoader(test_dataset,batch_size=conf["batch_size"],shuffle=True, pin_memory = pin), test_dataset
            else:
                print("Process the data of ", user_id)
                usr1 = char[sample_user[user_id]]
                usr2 = char[sample_user[user_id + conf["no_models"]]]
                x = data_conf["user_data"][usr1]["x"] + data_conf["user_data"][usr2]["x"]
                y = data_conf["user_data"][usr1]["y"] + data_conf["user_data"][usr2]["y"]

                data_x = [word_to_indices(word) for word in x]
                data_y = [letter_to_vec(c) for c in y]
                # data_x = torch.LongTensor(data_x)
                # data_y = torch.LongTensor(data_y)

                train_partition = int(len(data_x) * 0.8)

                train_data_x += data_x[:train_partition]
                train_data_y += data_y[:train_partition]

                train_data_x = torch.LongTensor(train_data_x)
                train_data_y = torch.LongTensor(train_data_y)

                train_dataset = Mydataset(train_data_x, train_data_y)
                print(user_id, "has data of ", len(train_data_x))
                return torch.utils.data.DataLoader(train_dataset, batch_size=conf["batch_size"], pin_memory = pin,drop_last=True)



    elif name == 'mnist':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
 
        train_dataset = datasets.EMNIST(dir, train=True, download=True, transform=transform_train,split = 'byclass' )
        eval_dataset = datasets.EMNIST(dir, train=False, transform=transform_test,split = 'byclass' )
    elif name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
 
        train_dataset = datasets.CIFAR10(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
    elif name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
 
        train_dataset = datasets.CIFAR100(dir, train=True, download=True, transform=transform_train)
        eval_dataset = datasets.CIFAR100(dir, train=False, transform=transform_test)
    
    
    if roll == "s":
        label_to_indices = {}
        eval_loader_list = {}

        for i in range(len(eval_dataset)):
            label = eval_dataset.targets[i]
            if label not in label_to_indices:
                label_to_indices[label] = []
            if len(label_to_indices[label]) < conf["server_eval_size"]:
                label_to_indices[label].append(i)

        
        for label, indices in label_to_indices.items():
            eval_loader_label = torch.utils.data.DataLoader(eval_dataset,batch_size=conf["batch_size"],pin_memory = pin,
                                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
            eval_loader_list[label] = eval_loader_label
        eval_loader = torch.utils.data.DataLoader(eval_dataset,batch_size=conf["batch_size"],shuffle=True, drop_last=True,pin_memory = pin)
        return eval_loader, eval_loader_list
    else:
        if user_id == 0:
            if conf["redistribution"] == "y":
                label_to_indices = {}
                eval_loader_list = {}

                for i in range(len(train_dataset)):
                    label = int(train_dataset.targets[i])
                    if label not in label_to_indices:
                        label_to_indices[label] = []
                    label_to_indices[label].append(i)

                Data_partition = {}
                if conf["non_iid"] == "HeteroDiri":
                    partition = np.random.dirichlet([conf["alpha"]]*conf["no_models"], len(label_to_indices))
                    label_partition = {}
                    for label, indices in label_to_indices.items():
                        label_partition[label] = partition[label]

                    label_indices = {}
                    for label, indices in label_to_indices.items():
                        label_indices[label] = np.arange(0, len(indices), 1)

                    Dataset_label_indices = {}
                    for label, parti in label_partition.items():
                        par = []
                        for p in parti:
                            par.append(round(p * len(label_indices[label])))
                        par = np.cumsum(par)
                        par[-1] = len(label_indices[label])
                        Dataset_label_indices[label] = np.split(label_indices[label], par)

                    for label in label_partition:
                        Data_partition[label] = []
                        for indices in Dataset_label_indices[label]:
                            data_indices = []
                            for i in indices:
                                data_indices.append(label_to_indices[label][i])
                            Data_partition[label].append(data_indices)

                elif conf["non_iid"] == "Shards":
                    squence_indeices = []
                    for label, indices in label_to_indices.items():
                        for i in indices:
                            squence_indeices.append(i)

                    squence = list(range(len(squence_indeices)))

                    step = int(len(squence)/(conf["no_models"] * conf["alpha"]))
                    shards = [squence[i:i+step] for i in range(0,len(squence),step)]

                    len_shards = list(range(len(shards)))

                    for i in range(conf["no_models"]):
                        id = random.sample(len_shards, conf["alpha"])
                        i_data = []
                        for j in range(conf["alpha"]):
                            i_data += shards[id[j]]
                            len_shards.remove(id[j])
                        s_data = []
                        for s in i_data:
                            s_data.append(squence_indeices[s])
                        Data_partition[i] = s_data

                elif conf["non_iid"] == "QuanSkew":
                    labels = list(range(len(label_to_indices)))

                    for i in range(conf["no_models"]):
                        client_labels = random.sample(labels, int(conf["alpha"]*len(labels)))
                        client_data = []
                        for cl in client_labels:
                            client_data += label_to_indices[cl]

                        Data_partition[i] = client_data
                elif conf["non_iid"] == "Unbalance_Diri":
                    n_class = conf["CLASS_NUM"]
                    partition = np.random.dirichlet([1]*n_class, 1) 
                    a = partition[0]
                    ma = max(a)
                    diri_data = []
                    for i in range(n_class):
                        diri_data.append(int((a[i]/ma)*len(label_to_indices[i])))
                    y0 = np.random.lognormal(0, conf["alpha"], conf["no_models"])
                    s = sum(diri_data)/sum(y0)
                    y = y0*s

                    point_indices = [0]*n_class
                    Data_partition = []
                    for i in range(conf["no_models"]):
                        n_label = []
                        c_partition = []
                        data_indices = []
                        for j in range(n_class):
                            n = int(a[j]*y[i])
                            n_label.append(n) 
                            c_partition.append([point_indices[j],point_indices[j]+n])
                            point_indices[j] += n
                            data_indices += label_to_indices[j][c_partition[j][0]: c_partition[j][1]] 
                        Data_partition.append(data_indices)

                elif conf["non_iid"] == "iid":
                    Data_partition = []
                    for i in range(conf["no_models"]):
                        data_indices = random.sample(list(range(0, len(train_dataset))), int(conf["alpha"]*len(train_dataset)))
                        print(len(data_indices))
                        Data_partition.append(data_indices)
                    else:
                        pass

                torch.save(Data_partition, "./data_partition_with_"+conf["non_iid"]+"_dataset_"+str(conf["type"])+"_"+str(conf["alpha"])+"_and_"+str(conf["no_models"])+"_models.pt")
            else:
                Data_partition = torch.load("./data_partition_with_"+conf["non_iid"]+"_dataset_"+str(conf["type"])+"_"+str(conf["alpha"])+"_and_"+str(conf["no_models"])+"_models.pt")
        else:
            Data_partition = torch.load("./data_partition_with_"+conf["non_iid"]+"_dataset_"+str(conf["type"])+"_"+str(conf["alpha"])+"_and_"+str(conf["no_models"])+"_models.pt")

        train_indices = []
        if conf["non_iid"] == "HeteroDiri":
            for label, indices in Data_partition.items():
                train_indices += indices[user_id]
        elif conf["non_iid"] == "Shards":
            train_indices = Data_partition[user_id]
        elif conf["non_iid"] == "QuanSkew":
            train_indices = Data_partition[user_id]
        elif conf["non_iid"] == "Unbalance_Diri":
            train_indices = Data_partition[user_id]
        elif conf["non_iid"] == "x":
            train_indices = Data_partition[user_id]
        else:
            pass

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf["batch_size"], pin_memory = pin,drop_last=True,
                                    sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices))
        return train_loader

if __name__ == "__main__":
    import json
    import time
    with open("tconf.json", 'r', encoding='utf-8') as f:
        conf = json.load(f)

    train_loader = get_dataset("../data/", conf["type"], "c", conf, 0)
    #eval_loader,e = get_dataset("../data/", conf["type"], "s", conf, 0)

    # for batch_id, batch in enumerate(train_loader):
    #     data, target = batch
    #     dataset_size = data.size()
    # print(dataset_size)


    # for batch_id, batch in enumerate(eval_loader):
    #     data, target = batch
    #     dataset_size = data.size()
    # print(dataset_size)

    
    # b = torch.load("./data_partition_with_QuanSkew_dataset_cifar10_0.5_and_100_models.pt")
    # transform_train = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ])
 
    # train_dataset = datasets.CIFAR10("../data", train=True, download=True, transform=transform_train)

    # for i in range(100):
    #     label_list = []
    #     indices = b[i]
    #     for id in indices:
    #         l = train_dataset.targets[id]
    #         if l not in label_list:
    #             label_list.append(l)
        
    #     print("Client:", i, "lables:", label_list, "data_size:", len(b[i]), "label_size:", len(label_list))
    #     time.sleep(5)