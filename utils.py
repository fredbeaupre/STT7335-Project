import glob
import os
import numpy as np
import csv
import yaml
import torch
from data import TabularBankDataset
import pandas as pd
from sklearn.model_selection import train_test_split


def yes_no_to_number(value):
    if value == "yes":
        return 1
    elif value == "no":
        return 0
    else:
        return np.nan


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SaveBestModel:
    def __init__(self, save_dir, metric_name, best_metric_val=float('inf'), maximize=True):
        self.best_metric_val = best_metric_val
        self.metric_name = metric_name
        self.save_dir = save_dir
        self.maximize = maximize

    def __call__(self, current_val, epoch, model, optimizer, criterion=None):
        if self.maximize:
            if current_val > self.best_metric_val:
                self.best_metric_val = current_val
                print(f"Best {self.metric_name}: {self.best_metric_val}")
                print(
                    f"Saving best model for epoch: {epoch + 1} at {self.save_dir}\n")
                torch.save({
                    "epoch": epoch+1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                }, "{}/best_model.pth".format(self.save_dir))
        else:
            if current_val < self.best_metric_val:
                self.best_metric_val = current_val
                print(f"Best {self.metric_name}: {self.best_metric_val}")
                print(
                    f"Saving best model for epoch: {epoch + 1} at {self.save_dir}\n")
                torch.save({
                    "epoch": epoch+1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                }, "{}/best_model.pth".format(self.save_dir))


def get_config_from_args(arg):
    if arg == "dummy":
        return "configs/dummy.yml"
    else:
        exit("Requested configuration does not exist")


def load_config(path, template=None):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load_csv_to_pandas(datapath="./bank-additional-full.csv", drop_na=True):
    if drop_na:
        df = pd.read_csv(datapath, header=0, sep=';').dropna()
    else:
        df = pd.read_csv(datapath, header=0, sep=';')
    return df


def create_dataloaders(path="./bank_additional_full.csv"):
    data = load_csv_to_pandas()
    train_data, val_data, test_data = split_dataset(data)
    test_data.to_csv(
        "./saved_models/data2vec_classification/test_data.csv", index=False)
    train_data.to_csv(
        "./saved_models/data2vec_classification/train_data.csv", index=False)
    train_set = TabularBankDataset(data=train_data)
    val_set = TabularBankDataset(data=val_data)
    test_set = TabularBankDataset(data=test_data)
    emb_dims_train = train_set.get_emb_dims()
    emb_dims_val = val_set.get_emb_dims()
    emb_dims_test = test_set.get_emb_dims()
    batch_size = 512
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    return (train_loader, val_loader, test_loader), (emb_dims_train, emb_dims_val, emb_dims_test)


def create_balanced_loaders(path="./bank_additional_full.csv"):
    data = load_csv_to_pandas()
    print(data.shape)
    yes_data = data[data["y"] == "yes"]
    no_data = data[data["y"] == "no"]
    N_yes = yes_data.shape[0]
    no_data = no_data.sample(n=N_yes)
    data = pd.concat([yes_data, no_data])
    print(data.shape)
    data = data.sample(frac=1)
    print(data.shape)
    train_data, val_data, test_data = split_dataset(data)
    test_data.to_csv(
        "./saved_models/data2vec_balanced_classification/test_data.csv", index=False)
    train_data.to_csv(
        "./saved_models/data2vec_balanced_classification/train_data.csv", index=False)
    train_set = TabularBankDataset(data=train_data)
    val_set = TabularBankDataset(data=val_data)
    test_set = TabularBankDataset(data=test_data)
    emb_dims_train = train_set.get_emb_dims()
    emb_dims_val = val_set.get_emb_dims()
    emb_dims_test = test_set.get_emb_dims()
    batch_size = 256
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    return (train_loader, val_loader, test_loader), (emb_dims_train, emb_dims_val, emb_dims_test)


def omniloader(path="./bank_additional_full.csv"):
    data = load_csv_to_pandas()
    dataset = TabularBankDataset(data=data)
    emb_dims = dataset.get_emb_dims()
    batch_size = 512
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    return dataloader, emb_dims


def split_dataset(dataset):
    N_train = dataset.shape[0]
    N_test = int(N_train * 0.20)
    N_val = int(N_train * 0.30)

    # sample the test set
    test_set = dataset.sample(n=N_test)
    # remove test set from train set
    train_set = pd.concat([dataset, test_set]).drop_duplicates(keep=False)
    # sampe validation set
    val_set = train_set.sample(n=N_val)
    train_set = pd.concat([train_set, val_set]).drop_duplicates(keep=False)
    return train_set, val_set, test_set
