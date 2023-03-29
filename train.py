import numpy as np
import matplotlib.pyplot as plt
import glob
import utils
import models
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str)
parser.add_argument("--model", required=True, type=str)
args = parser.parse_args()
CONFIG_FILE = utils.get_config_from_args(args.config)
config = utils.load_config(CONFIG_FILE)
print("\nconfig\n")

def validation_step():
    pass

def train_step():
    pass

def train():
    pass

def main():
    pass

if __name__=="__main__":
    main()
