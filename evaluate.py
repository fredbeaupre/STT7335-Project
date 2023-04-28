import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import utils
import models
from data2vec import Data2Vec
from tqdm import tqdm
import metrics


def compute_metrics(model, dataloader, device):
    model.eval()
    all_preds = []
    raw_preds = []
    all_labels = []
    for y, cont_x, cat_x in tqdm(dataloader, desc="Computing metrics..."):
        y, cont_x, cat_x = y.to(device), cont_x.to(device), cat_x.to(device)
        raw_yhat = model(cont_x, cat_x, task="classification")
        yhat = torch.tensor(raw_yhat > 0.5)
        all_preds.append(yhat)
        raw_preds.append(raw_yhat)
        all_labels.append(y)
    all_preds = torch.cat(all_preds, dim=0).cpu().detach().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().detach().numpy()
    raw_preds = torch.cat(raw_preds, dim=0).cpu().detach().numpy()
    accuracy = metrics.compute_accuracy(all_preds, all_labels)
    precision = metrics.compute_precision(all_preds, all_labels)
    recall = metrics.compute_recall(all_preds, all_labels)
    print(f"Accuracy = {accuracy}\nPrecision = {precision}\nRecall = {recall}")
    metrics.compute_roc(raw_preds, all_labels)
    lifts, response_rates = metrics.compute_lift(raw_preds, all_labels)
    for res, lf in zip(response_rates, lifts):
        print(res, lf)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, emb_dims = utils.create_dataloaders(drop_na=True)
    train_loader, val_loader, test_loader = loaders
    emb_dims_train, _, emb_dims_test = emb_dims
    emb_dims_train[0] = (12, 6)
    emb_dims_train[1] = (4, 2)
    emb_dims_train[2] = (8, 4)
    emb_dims_train[3] = [3, 2]
    emb_dims_train[4] = (3, 2)
    emb_dims_train[5] = (3, 2)
    for dim in emb_dims:
        print(dim)
    emb_dims_train[3] = (3, 2)
    encoder = models.FeedForwardNet(
        emb_dims=emb_dims_train,
        num_continuous=8,
        lin_layer_sizes=[50, 100],
        output_size=16,
        emb_dropout=0.04,
        lin_layer_dropouts=[0.001, 0.01]
    ).to(device)
    model = Data2Vec(encoder=encoder, device=device).to(device)
    checkpoint = torch.load(
        "./saved_models/data2vec_classif_onlyfull_imbalanced/best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    compute_metrics(model, test_loader, device)


if __name__ == "__main__":
    main()
