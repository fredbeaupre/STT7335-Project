import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
import models
from data2vec import Data2Vec
from tqdm import tqdm
import pandas as pd


def impute(model, dataloader, device):
    model.eval()
    imputed_dataset = []
    for y, cont_x, cat_x in tqdm(dataloader, desc="Validation..."):
        y, cont_x, cat_x = y.to(device), cont_x.to(device), cat_x.to(device)
        recon, _ = model(cont_x, cat_x, task="reconstruction")
        imputed_dataset.append(recon)
    imputed_dataset = torch.cat(imputed_dataset).cpu().detach().numpy()
    imputed_dataset = pd.DataFrame(imputed_dataset.numpy())
    imputed_dataset.to_csv(
        "./saved_models/data2vec_imputation/imputed_dataset")


def conclude(save_dict):
    torch.save(
        save_dict, "./saved_models/data2vec_imputation/final_decoder.pth")
    loss = save_dict["loss"]
    val_loss = save_dict["val_loss"]
    epochs = np.arange(0, len(loss), 1)
    fig = plt.figure(1)
    plt.plot(epochs, loss, color='tab:blue', label="Train", marker='.')
    plt.plot(epochs, val_loss, color='tab:orange',
             label='Valid.', marker='.')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig("./saved_models/data2vec_imputation/loss_fig_decoder.png")
    fig.savefig("./saved_models/data2vec_imputation/loss_fig_decoder.pdf",
                bbox_inches='tight', transparent=True)


def main():
    num_epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader, emb_dims = utils.load_missing()
    encoder = models.FeedForwardNet(
        emb_dims=emb_dims,
        num_continuous=8,
        lin_layer_sizes=[50, 100],
        output_size=16,
        emb_dropout=0.04,
        lin_layer_dropouts=[0.001, 0.01]
    ).to(device)

    model = Data2Vec(
        encoder=encoder, device=device
    ).to(device)

    checkpoint = torch.load(
        "./saved_models/data2vec_imputation/best_decoder.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    impute(model, dataloader, device)


if __name__ == "__main__":
    main()
