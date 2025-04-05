import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Number of data points in our set

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

points = 52069
train_size = int(0.8 * points)


class TempAnomalyDataset(Dataset):
    def __init__(self, file_path, lookback=8):
        self.data = pd.read_json(file_path)
        self.data = self.data.to_numpy()
        self.data_in = []
        self.data_out = []
        for i in range(len(self.data) - lookback):
            feature = self.data[i:i+lookback, 4]
            target = self.data[i+1:i+lookback+1, 4]
            self.data_in.append(feature)
            self.data_out.append(target)
        self.data_in = np.array(self.data_in)
        self.data_out = np.array(self.data_out)
        self.data_in = torch.tensor(self.data_in)
        self.data_out = torch.tensor(self.data_out)

    def __len__(self):
        return points

    def __getitem__(self, idx):
        if idx >= points:
            idx = points - 1
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data_in[idx], self.data_out[idx]


anomaly_dataset = TempAnomalyDataset("data/berkeley/data.json", lookback=8)
train_data, valid_data = torch.utils.data.random_split(
    anomaly_dataset, [train_size, points - train_size])


# Hack to be able to put LSTMCells together
class sanitize(nn.Module):
    def forward(self, x):
        tensor, _ = x
        return tensor


class TempAnomalyNetwork(nn.Module):

    def __init__(self, layers=5, lookback=8):
        super().__init__()
        L = []
        i = 1
        for _ in range(layers):
            i *= 2
            L.append(nn.LSTMCell((i // 2) * lookback, i * lookback, dtype=float))
            L.append(sanitize())
        self.network = nn.Sequential(*L)
        self.classifier = nn.Linear(i * lookback, lookback, dtype=float)

    def forward(self, x):
        x = self.network(x)
        x = self.classifier(x)
        return x


def train(lr=1e-3, reg=1e-3, epochs=10, batch_size=8, lookback=8, weight_decay=0):
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size)

    model = TempAnomalyNetwork(layers=2, lookback=lookback).to(device)
    opt = optim.Adam(model.parameters(), lr=lr,
                     weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for i in range(epochs):
        model.train()
        for batch_xs, batch_ys in train_loader:
            batch_xs = batch_xs.to(device)
            batch_ys = batch_ys.to(device)
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


model = train()
