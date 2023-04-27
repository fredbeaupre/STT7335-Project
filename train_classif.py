import numpy as np
import matplotlib.pyplot as plt
import utils
import torch
import models
from tqdm import tqdm
from data2vec import Data2Vec


def compute_accuracy(preds, labels):
    assert preds.shape == labels.shape
    correct = torch.sum(preds == labels)
    return correct.item() / preds.shape[0]


def validation_step(model, validation_loader, criterion, device):
    model.eval()
    val_loss = utils.AverageMeter()
    all_preds = []
    all_labels = []
    for y, cont_x, cat_x in tqdm(validation_loader, desc="Validation"):
        y, cont_x, cat_x = y.to(device), cont_x.to(device), cat_x.to(device)
        yhat = model(cont_x, cat_x, task="classification")
        loss = criterion(yhat, y)
        val_loss.update(val=loss.item(), n=cont_x.shape[0])
        yhat = torch.tensor(yhat > 0.5)
        all_preds.append(yhat)
        all_labels.append(y)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    acc = compute_accuracy(all_preds, all_labels)
    return val_loss.avg, acc


def train_step(model, train_loader, criterion, optimizer, device):
    loss_meter = utils.AverageMeter()
    model.train()
    all_preds = []
    all_labels = []
    for y, cont_x, cat_x in tqdm(train_loader, desc="Training..."):
        y, cont_x, cat_x = y.to(device), cont_x.to(device), cat_x.to(device)
        optimizer.zero_grad()
        yhat = model(cont_x, cat_x, task="classification")
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()
        loss_meter.update(val=loss.item(), n=cont_x.shape[0])
        yhat = torch.tensor(yhat > 0.5)
        all_preds.append(yhat)
        all_labels.append(y)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    acc = compute_accuracy(all_preds, all_labels)
    return loss_meter.avg, acc


def train(model, train_loader, validation_loader, optimizer, criterion, num_epochs, device):
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    save_best_model = utils.SaveBestModel(save_dir="./saved_models/data2vec_classification",
                                          metric_name="validation_accuracy", best_metric_val=-1, maximize=True)
    for epoch in range(num_epochs):
        t_loss, t_acc = train_step(
            model, train_loader, criterion, optimizer, device)
        train_loss.append(t_loss)
        train_acc.append(t_acc)
        print(f"---> Epoch {epoch + 1} training loss: {t_loss}")
        print(f"---> Epoch {epoch + 1} training accuracy: {t_acc}")
        v_loss, v_acc = validation_step(
            model, validation_loader, criterion, device)
        val_loss.append(v_loss)
        val_acc.append(v_acc)
        print(f"---> Epoch {epoch + 1} validation loss: {v_loss}")
        print(f"---> Epoch {epoch + 1} validation accuracy: {v_acc}")
        save_best_model(v_acc, epoch, model, optimizer, criterion=criterion)
    return (train_loss, train_acc), (val_loss, val_acc)


def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for y, cont_x, cat_x in tqdm(test_loader, desc="Testing..."):
        y, cont_x, cat_x = y.to(device), cont_x.to(device), cat_x.to(device)
        yhat = model(cont_x, cat_x, task="classification")
        yhat = torch.tensor(yhat > 0.5)
        all_preds.append(yhat)
        all_labels.append(y)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    acc = compute_accuracy(all_preds, all_labels)
    return acc


def conclude(save_dict):
    torch.save(save_dict, "./saved_models/data2vec_classification/final_model.pth")
    loss = save_dict["loss"]
    val_loss = save_dict["val_loss"]
    acc = save_dict["acc"]
    val_acc = save_dict["val_acc"]
    epochs = np.arange(0, len(loss), 1)
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(epochs, loss, color="tab:blue", label="Train", marker='.')
    axs[0].plot(epochs, val_loss, color="tab:orange",
                label="Valid.", marker=".")
    axs[1].plot(epochs, acc, color="tab:blue", label="Train", marker='.')
    axs[1].plot(epochs, val_acc, color='tab:orange', label="Valid", marker='.')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[0].set_ylabel("Loss")
    plt.legend()
    fig.savefig("./saved_models/data2vec_classification/loss_fig.png",
                bbox_inches="tight")
    fig.savefig("./saved_models/data2vec_classification/loss_fig.pdf",
                bbox_inches="tight", transparent=True)


def main():
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, emb_dims = utils.create_dataloaders()
    train_loader, val_loader, test_loader = loaders
    emb_dims_train, _, _ = emb_dims
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
        "./saved_models/data2vec_distillation/best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.00001)

    (train_loss, train_acc), (val_loss, val_acc) = train(
        model, train_loader, val_loader, optimizer, criterion, num_epochs, device)

    save_dict = {
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": train_loss,
        "val_loss": val_loss,
        "acc": train_acc,
        "val_acc": val_acc,
    }
    conclude(save_dict)

    # TODO: reload best checkpoint and evaluate
    eval_checkpoint = torch.load(
        "./saved_models/data2vec_classification/best_model.pth")
    model.load_state_dict(eval_checkpoint["model_state_dict"])
    test_accuracy = evaluate(model, test_loader, device)
    print("Accuracy on the test set = {}".format(round(test_accuracy, 3)))


if __name__ == "__main__":
    main()
