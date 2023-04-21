# Test network

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from poutyne import set_seeds, Model, plot_history, ReduceLROnPlateau
from matplotlib import pyplot as plt
from datetime import datetime

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
print("---Finished loading data\n")
# Todo drop na
# Todo Régularization ?

# Create Dataloaders
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)

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
