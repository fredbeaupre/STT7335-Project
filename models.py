import torch
import numpy as np


class FeedForwardNet(torch.nn.Module):
    def __init__(
            self,
            emb_dims,
            num_continuous,
            lin_layer_sizes,
            output_size,
            emb_dropout,
            lin_layer_dropouts
    ):
        super().__init__()

        # Embedding Layers
        self.embedding_layers = torch.nn.ModuleList([
            torch.nn.Embedding(x, y) for x, y in emb_dims
        ])

        num_embeds = sum([y for x, y in emb_dims])
        self.num_embeds = num_embeds
        self.num_continuous = num_continuous

        # Linear Layers
        first_linear = torch.nn.Linear(
            self.num_embeds + self.num_continuous, lin_layer_sizes[0]
        )

        self.lin_layers = torch.nn.ModuleList(
            [first_linear] + [
                torch.nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i+1]) for i in range(len(lin_layer_sizes) - 1)
            ]
        )

        for lin_layer in self.lin_layers:
            torch.nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output layer
        self.output_layer = torch.nn.Linear(lin_layer_sizes[-1], output_size)
        torch.nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm layers
        self.first_batchnorm = torch.nn.BatchNorm1d(self.num_continuous)
        self.batch_norms = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(size) for size in lin_layer_sizes]
        )

        # Dropout layers
        self.emb_dropout_layer = torch.nn.Dropout(emb_dropout)
        self.dropout_layers = torch.nn.ModuleList(
            [torch.nn.Dropout(size) for size in lin_layer_dropouts]
        )

    def forward(self, cont_data, cat_data):
        if self.num_embeds != 0:
            x = [emb_layer(cat_data[:, i])
                 for i, emb_layer in enumerate(self.embedding_layers)]
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)

        if self.num_continuous != 0:
            normalized_cont_data = self.first_batchnorm(cont_data)

            if self.num_embeds != 0:
                x = torch.cat([x, normalized_cont_data], 1)
            else:
                x = normalized_cont_data

        for lin_layer, dropout_layer, bn_layer in zip(
            self.lin_layers, self.dropout_layers, self.batch_norms
        ):
            x = torch.nn.functional.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)
        x = self.output_layer(x)
        return x
