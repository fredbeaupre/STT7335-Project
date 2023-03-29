import glob
import os
import numpy as np
import csv
import yaml

class SaveBestModel:
    def __init__(self, save_dir, metric_name, best_metric_val=float('inf'), maximize=True):
        self.best_metric_val = best_metric_val
        self.metric_name = metric_name
        self.save_dir = save_dir

    def __call__(self, current_val, epoch, model, optimizer, criterion=None):
        if maximize:
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
