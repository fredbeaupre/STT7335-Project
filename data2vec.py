import torch
from ema import EMA


class Data2Vec(torch.nn.Module):
    def __init__(self, encoder, device):
        super(Data2Vec, self).__init__()
        self.embed_dim = 768
        self.encoder = encoder
        self.device = device
        self.ema = EMA(self.encoder, self.device, decay=0.999)
        self.ema_decay = 0.999
        self.ema_end_decay = 0.9999
        self.ema_anneal_end_step = 300000

        self.fc = torch.nn.Linear(16, 1)

        # self.decoder = torch.nn.Sequential(
        #     torch.nn.Linear(16, 100),
        #     torch.nn.ReLU(),
        #     torch.nn.BatchNorm1d(100),
        #     torch.nn.Linear(100, 50),
        #     torch.nn.ReLU(),
        #     torch.nn.BatchNorm1d(50),
        #     torch.nn.Linear(50, 37),
        #     torch.nn.ReLU(),
        #     torch.nn.BatchNorm1d(37),
        #     torch.nn.Linear(37, 18)
        # )

        # self.decoder = torch.nn.Linear(16, 50)

    def ema_step(self):
        if self.ema_decay != self.ema_end_decay:
            if self.ema.num_updates >= self.ema_anneal_end_step:
                decay = self.ema_end_decay
            else:
                decay = self.ema.get_annealed_rate(
                    self.ema_decay,
                    self.ema_end_decay,
                    self.ema.num_updates,
                    self.ema_anneal_end_step
                )
            self.ema_decay = decay
        if self.ema_decay < 1:
            self.ema.step(self.encoder)

    def forward(self, cont_x, cat_x, target=None, task="distillation"):

        if task == "classification":
            # TODO: add classification head
            x = self.encoder(cont_x, cat_x, mask=False)
            x = torch.sigmoid(self.fc(x))
            return x

        # elif task == "reconstruction":
        #     # TODO: reconstruction decoder
        #     target = torch.cat((cont_x, cat_x), axis=1)
        #     print(cont_x.shape, cat_x.shape)
        #     x = self.encoder(cont_x, cat_x, mask=True)
        #     x = self.decoder(x)
        #     return x, target

        elif task == "distillation":
            x = self.encoder(cont_x, cat_x, mask=True)
            with torch.no_grad():
                self.ema.model.eval()
                y = self.ema.model(cont_x, cat_x)
            return x, y
        else:
            exit("The required task cannot be performed")
