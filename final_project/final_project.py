import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Number of data points in our set


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

points = 52077
train_size = int(0.8 * points)
lookback = 2000


class TempAnomalyDataset(Dataset):
    def __init__(self, file_path, lookback=8):
        self.data = pd.read_json(file_path)
        self.data = self.data.to_numpy()
        self.data_in = []
        self.data_out = []
        self.lookback = lookback
        for i in range(len(self.data) - self.lookback):
            feature = self.data[i: i + self.lookback, 4]
            target = self.data[i + 1, 4]
            self.data_in.append(feature)
            self.data_out.append(target)
        self.data_in = np.array(self.data_in)
        self.data_out = np.array(self.data_out)
        self.data_in = torch.tensor(self.data_in)
        self.data_out = torch.tensor(self.data_out)

    def __len__(self):
        return points

    # Kinda hacky solution but it works for now
    def __getitem__(self, idx):
        if idx >= points - self.lookback:
            idx = points - self.lookback - 1
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data_in[idx], self.data_out[idx]


anomaly_dataset = TempAnomalyDataset(
    "data/berkeley/data.json", lookback=lookback)
train_data, valid_data = torch.utils.data.random_split(
    anomaly_dataset, [train_size, points - train_size]
)


# We want this network to output log probabilities over possible continuations
# of the sequence if we are going to implement beam search
class TempAnomalyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Has output size (32, 400)
            nn.Conv1d(
                in_channels=8,
                out_channels=32,
                kernel_size=100,
                stride=5,
                padding=49,
                dtype=float,
            ),
            nn.ReLU(),
            # Has output size (64, 200)
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=20,
                stride=2,
                padding=9,
                dtype=float,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=20,
                stride=2,
                padding=9,
                dtype=float,
            ),
        )

    def forward(self, x):
        x = self.network(x)
        print(x.shape)
        return x


def beam_search(model, input, k):


def train(lr=1e-3, reg=1e-3, epochs=10, batch_size=8, lookback=8, weight_decay=0):
    train_loader = DataLoader(train_data, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    model = TempAnomalyNetwork().to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for i in range(epochs):
        model.train()
        for batch_xs, batch_ys in train_loader:
            batch_xs = batch_xs.to(device)
            batch_ys = batch_ys.to(device)
            print(batch_xs.shape)
            # Batch_xs has shape (batch_size, lookback)
            pred = model(batch_xs)
            loss = loss_fn(pred, batch_ys)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        losses = []
        for batch_xs, batch_ys in valid_loader:
            batch_xs = batch_xs.to(device)
            batch_ys = batch_ys.to(device)
            pred = model(batch_xs)
            loss = loss_fn(pred, batch_ys)
            losses.append(loss.item())
        losses = np.array(losses, dtype=float)
        print("Epoch: ", i, "Average validation loss: ", np.mean(losses))

    return model


model = train(lookback=lookback)
