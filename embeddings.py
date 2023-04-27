import numpy as np
import matplotlib.pyplot as plt
import utils
import models
import torch
import data
import models
from data2vec import Data2Vec
from tqdm import tqdm


def save_embeddings(dataloader, model, device):
    embeddings = []
    for y, cont_x, cat_x in tqdm(dataloader, desc="Generating embeddings..."):
        y, cont_x, cat_x = y.to(device), cont_x.to(device), cat_x.to(device)
        _, full_pred = model(cont_x, cat_x)
        embeddings.append(full_pred)
    embeddings = torch.cat(embeddings)
    embeddings = embeddings.cpu().detach().numpy()
    np.savez("./embeddings", embeddings=embeddings)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader, emb_dims = utils.omniloader()
    encoder = models.FeedForwardNet(
        emb_dims=emb_dims,
        num_continuous=8,
        lin_layer_sizes=[50, 100],
        output_size=16,
        emb_dropout=0.04,
        lin_layer_dropouts=[0.001, 0.01]
    ).to(device)

    model = Data2Vec(encoder=encoder, device=device).to(device)

    checkpoint = torch.load(
        "./saved_models/data2vec_distillation/best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    for p in model.parameters():
        print(p)
    save_embeddings(dataloader, model, device)


if __name__ == "__main__":
    main()
