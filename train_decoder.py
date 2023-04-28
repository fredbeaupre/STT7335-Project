import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
import models
from data2vec import Data2Vec
from tqdm import tqdm


def validation_step(model, validation_loader, criterion, device):
    model.eval()
    val_loss = utils.AverageMeter()
    for y, cont_x, cat_x in tqdm(validation_loader, desc="Validation..."):
        y, cont_x, cat_x = y.to(device), cont_x.to(device), cat_x.to(device)
        recon, target = model(cont_x, cat_x, task="reconstruction")
        loss = criterion(recon, target)
        val_loss.update(val=loss.item(), n=cont_x.shape[0])
    return val_loss.avg


def train_step(model, train_loader, criterion, optimizer, device):
    loss_meter = utils.AverageMeter()
    model.train()
    for y, cont_x, cat_x in tqdm(train_loader, desc="Training..."):
        y, cont_x, cat_x = y.to(device), cont_x.to(device), cat_x.to(device)
        optimizer.zero_grad()
        recon, target = model(cont_x, cat_x, task="reconstruction")
        print(recon[0])
        print(target[0])
        print(recon.shape, target.shape)
        loss = criterion(recon, target)
        loss.backward()
        optimizer.step()
        loss_meter.update(val=loss.item(), n=cont_x.shape[0])

    return loss_meter.avg


def train(model, train_loader, validation_loader, optimizer, criterion, num_epochs, device):
    train_loss, val_loss = [], []
    save_best_model = utils.SaveBestModel(save_dir="./saved_models/data2vec_imputation",
                                          metric_name="validation_loss", best_metric_val=np.Inf, maximize=False)
    for epoch in range(num_epochs):
        t_loss = train_step(model, train_loader, criterion, optimizer, device)
        model.ema_step()
        train_loss.append(t_loss)
        print(f"---> Epoch {epoch + 1} training loss: {t_loss}")
        v_loss = validation_step(model, validation_loader, criterion, device)
        val_loss.append(v_loss)
        print(f"---> Epoch {epoch + 1} validation loss: {v_loss}")
        save_best_model(v_loss, epoch, model, optimizer, criterion=criterion)
    return train_loss,  val_loss


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
    loaders, emb_dims = utils.create_dataloaders()
    train_loader, val_loader, test_loader = loaders
    emb_dims_train, emb_dims_val, emb_dims_test = emb_dims
    encoder = models.FeedForwardNet(
        emb_dims=emb_dims_train,
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
        "./saved_models/data2vec_imputation/best_encoder.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.00001)

    train_loss, val_loss = train(model, train_loader, val_loader,
                                 optimizer, criterion, num_epochs, device)

    save_dict = {
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": train_loss,
        "val_loss": val_loss
    }
    conclude(save_dict)


if __name__ == "__main__":
    main()
