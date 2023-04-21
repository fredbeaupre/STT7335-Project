import torch
import torch.nn as nn
import numpy as np


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, hidden_layers_encoder, hidden_layers_decoder, size_embedding, num_classes=None):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Linear(input_dim, hidden_layers_encoder[0]))
        for i in range(len(hidden_layers_encoder)-1):
            self.encoder.append(nn.Linear(hidden_layers_encoder[i], hidden_layers_encoder[i+1]))
        self.encoder.append(nn.Linear(hidden_layers_encoder[-1], size_embedding))

        self.decoder = nn.ModuleList()
        self.decoder.append(nn.Linear(size_embedding, hidden_layers_decoder[0]))
        for i in range(len(hidden_layers_decoder)-1):
            self.decoder.append(nn.Linear(hidden_layers_decoder[i], hidden_layers_decoder[i+1]))
        self.decoder.append(nn.Linear(hidden_layers_decoder[-1], input_dim))

        self.relu = nn.ReLU()

        self.mode = "decoder"

    def forward(self, x):
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            if i < len(self.encoder)-1:
                x = self.relu(x)
        if self.mode == "decoder":
            for i in range(len(self.decoder)):
                x = self.decoder[i](x)
                if i < len(self.decoder) - 1:
                    x = self.relu(x)
        return x

    def enable_pred_embedding(self):
        self.mode = "embedding"

    def enable_pred_decoder(self):
        self.mode = "decoder"
