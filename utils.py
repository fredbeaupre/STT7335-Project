import glob
import os
import numpy as np
import csv
import yaml
import torch
from data import TabularBankDataset
import pandas as pd


def yes_no_to_number(value):
    if value == "yes":
        return 1
    elif value == "no":
        return 0
    else:
        return np.nan


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


def create_dataloader(path="./bank_additional_full.csv"):
    data = load_csv_to_pandas()
    dataset = TabularBankDataset(
        data=data,
    )
    emb_dims = dataset.get_emb_dims()
    batch_size = 512
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    return trainloader, emb_dims
