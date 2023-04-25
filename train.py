import numpy as np
import matplotlib.pyplot as plt
import utils
import torch
import models
from tqdm import tqdm

# parser = argparse.ArgumentParser()
# parser.add_argument("--config", required=True, type=str)
# parser.add_argument("--model", required=True, type=str)
# args = parser.parse_args()
# CONFIG_FILE = utils.get_config_from_args(args.config)
# config = utils.load_config(CONFIG_FILE)
# print("\nconfig\n")


def validation_step():
    pass


def train_step():
    pass


def train():
    pass


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, emb_dims = utils.create_dataloader()
    model = models.FeedForwardNet(
        emb_dims=emb_dims,
        num_continuous=8,
        lin_layer_sizes=[50, 100],
        output_size=1,
        emb_dropout=0.04,
        lin_layer_dropouts=[0.001, 0.01]
    ).to(device)

    for _ in range(5):
        for y, cont_x, cat_x in tqdm(trainloader, desc="Training..."):
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            y = y.to(device)

            preds = model(cont_x, cat_x)
            print(preds)


if __name__ == "__main__":
    main()
