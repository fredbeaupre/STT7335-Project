# Test network
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from poutyne import set_seeds, Model, plot_history, ReduceLROnPlateau
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np

from data import BankDataset
from models import FeedForwardNet


set_seeds(42)

# Training parameters
batch_size = 1024
nb_epochs = 150
lr = 0.01
patience = 20

# Model characteristics
encoder_dim = (12, 6)
embedding_dim = 2
decoder_dim = (6, 12)
input_dim = 23


# Load the data
print("---Start loading data")
train_data = BankDataset(path="bank_additional_clean_train.csv", target_input=True, mask_input=True, drop_na=True)
train_scaler = train_data.scaler
valid_data = BankDataset(path="bank_additional_clean_valid.csv", target_input=True, mask_input=True, drop_na=True,
                         scaler=train_scaler)
test_data = BankDataset(path="bank_additional_clean_test.csv", target_input=True, mask_input=True, drop_na=True,
                        scaler=train_scaler)
print("---Finished loading data\n")

# Create Dataloaders
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


# Check if GPU is available. If not, uses the CPU
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

# Creating the network
full_network = FeedForwardNet(input_dim=input_dim, hidden_layers_encoder=encoder_dim, hidden_layers_decoder=decoder_dim,
                              size_embedding=embedding_dim)

# Optimizer
optimizer = optim.Adam(full_network.parameters(), lr)
reduce_lr = ReduceLROnPlateau(patience=patience, monitor='val_loss', verbose=True)

# Loss function
loss_function = torch.nn.MSELoss()

# Using the Poutyne Model class to simplify training
model = Model(full_network, optimizer, loss_function=loss_function,
              epoch_metrics=[],
              device=device
              )

print("---Start training")
start_training = datetime.now()
history = model.fit_generator(train_loader, valid_loader, epochs=nb_epochs, verbose=True, callbacks=[reduce_lr])
end_training = datetime.now()
print(f"---Finished training: time elapsed :{end_training-start_training}\n")

# Plot training history
(figs, axes) = plot_history(history, show=False)
ax = axes[1]
plt.sca(ax)
# plt.ylim(top=20)
plt.ylabel("Loss")

plt.show()
# plt.savefig("training_curve.jpeg", format="jpeg", dpi=500)


print("---Start predictions")
# Predictions of original data where NA have been removed
# Used to compare fake missing data to original as masks are still applied
pred_original_NA_rm = True
if pred_original_NA_rm is True:
    pred_train, true_train = model.predict_generator(train_loader, return_ground_truth=True)
    pred_valid, true_valid = model.predict_generator(valid_loader, return_ground_truth=True)
    pred_test, true_test = model.predict_generator(test_loader, return_ground_truth=True)

    # Save predictions to file with ground truth
    colnames = list(train_data.bank_data.columns.values)
    for col in train_data.bank_data.columns:
        new_name = col + "_pred"
        colnames.append(new_name)

    df_embedding_train = pd.DataFrame(np.concatenate((true_train, pred_train), axis=1), columns=colnames)
    df_embedding_valid = pd.DataFrame(np.concatenate((true_valid, pred_valid), axis=1), columns=colnames)
    df_embedding_test = pd.DataFrame(np.concatenate((true_test, pred_test), axis=1), columns=colnames)

    df_embedding_train.to_csv("compare_predictions_train.csv")
    df_embedding_valid.to_csv("compare_predictions_valid.csv")
    df_embedding_test.to_csv("compare_predictions_test.csv")


# Using masks on the missing data only, predicts the original data and also the embedding
# Used to plot embeddings and to make predictive models from reconstructed data
pred_real_data = True
if pred_real_data is True:
    # New datasets including real NA and targeting real target values
    train_data = BankDataset(path="bank_additional_clean_train.csv", target_input=False, mask_input=False, drop_na=False,
                             scaler=train_scaler)
    valid_data = BankDataset(path="bank_additional_clean_valid.csv", target_input=False, mask_input=False, drop_na=False,
                             scaler=train_scaler)
    test_data = BankDataset(path="bank_additional_clean_test.csv", target_input=False, mask_input=False, drop_na=False,
                            scaler=train_scaler)

    # Corresponding dataloaders
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # Predict original data including NA
    pred_train, true_train = model.predict_generator(train_loader, return_ground_truth=True)
    pred_valid, true_valid = model.predict_generator(valid_loader, return_ground_truth=True)
    pred_test, true_test = model.predict_generator(test_loader, return_ground_truth=True)

    # Save predictions to file with ground truth
    colnames = list(train_data.bank_data.columns.values)
    colnames.append("y")

    df_embedding_train = pd.DataFrame(np.concatenate((pred_train, true_train), axis=1), columns=colnames)
    df_embedding_valid = pd.DataFrame(np.concatenate((pred_valid, true_valid), axis=1), columns=colnames)
    df_embedding_test = pd.DataFrame(np.concatenate((pred_test, true_test), axis=1), columns=colnames)

    df_embedding_train.to_csv("predictions_train.csv")
    df_embedding_valid.to_csv("predictions_valid.csv")
    df_embedding_test.to_csv("predictions_test.csv")

    # Change model mode to predict embedding instead of target
    model.network.enable_pred_embedding()

    # Predict embedding
    embedding_train, y_train = model.predict_generator(train_loader, return_ground_truth=True)
    embedding_valid, y_valid = model.predict_generator(valid_loader, return_ground_truth=True)
    embedding_test, y_test = model.predict_generator(test_loader, return_ground_truth=True)

    # Save embedding to file with y
    colnames = ["emb1", "emb2", "y"]
    df_embedding_train = pd.DataFrame(np.concatenate((embedding_train, y_train), axis=1), columns=colnames)
    df_embedding_valid = pd.DataFrame(np.concatenate((embedding_valid, y_valid), axis=1), columns=colnames)
    df_embedding_test = pd.DataFrame(np.concatenate((embedding_test, y_test), axis=1), columns=colnames)

    df_embedding_train.to_csv("embedding_train.csv")
    df_embedding_valid.to_csv("embedding_valid.csv")
    df_embedding_test.to_csv("embedding_test.csv")

print("---Finished predictions")
