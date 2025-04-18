import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.tensorboard as tb


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

points = 52077
train_size = int(0.8 * points)
seq_len = 2048
logdir = "logs"


class TempAnomalyDataset(Dataset):
    def __init__(self, file_path, seq_len):
        self.data = pd.read_json(file_path)
        self.data = self.data.to_numpy()
        self.data = torch.tensor(self.data[:, 4])
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        # Out of bound indices
        if idx > len(self.data):
            idx = len(self.data) - 1
        elif idx < self.seq_len:
            idx = self.seq_len
        return self.data[idx - self.seq_len : idx], self.data[
            idx - self.seq_len + 1 : idx + 1
        ]


anomaly_dataset = TempAnomalyDataset("~/scratch/data.json", seq_len=seq_len)
train_data, valid_data = torch.utils.data.random_split(
    anomaly_dataset, [train_size, len(anomaly_dataset) - train_size]
)

df = pd.read_json("data/berkeley/data.json")
time_series = df[["anomaly"]].values.astype("float32")
train_plot = np.ones_like(time_series) * np.nan
plt.plot(time_series, c="b")


class TempAnomalyNetwork(nn.Module):
    def __init__(self, layers=[32, 64, 128, 256]):
        super().__init__()
        c = 1
        L = []
        total_dilation = 1
        for l in layers:
            L.append(nn.ConstantPad1d((2 * total_dilation, 0), 0))
            L.append(nn.Conv1d(c, l, 3, dilation=total_dilation, dtype=float))
            L.append(nn.ReLU())
            total_dilation *= 2
            c = l
        self.network = nn.Sequential(*L)
        self.classifier = nn.Linear(layers[len(layers) - 1], 1, dtype=float)

    def forward(self, x):
        # Adding our features so convolutions don't get mad
        x = torch.unsqueeze(x, 1)
        # Feeding our data into our network
        x = self.network(x)
        # Moving the features to the back so we can take the linear properly
        x = torch.transpose(x, 1, 2)
        # Classifying our features
        x = self.classifier(x)
        # Removing size 1 dimension where our features were
        x = torch.squeeze(x, 2)
        return x


def train(lr=1e-3, epochs=10, batch_size=8, weight_decay=0.0):
    train_loader = DataLoader(train_data, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    logger = tb.SummaryWriter(
        logdir
        + "/"
        + "lr-"
        + str(lr)
        + "-wght-"
        + str(weight_decay)
        + "-arch-"
        + str(512)
    )

    model = TempAnomalyNetwork(layers=[32, 64, 128, 256]).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    global_step = 0

    for _ in range(epochs):
        model.train()
        for batch_xs, batch_ys in train_loader:
            batch_xs = batch_xs.to(device)
            batch_ys = batch_ys.to(device)
            pred = model(batch_xs)
            loss = loss_fn(pred, batch_ys)
            logger.add_scalar("train_loss", loss, global_step=global_step)
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1
        model.eval()
        val_step = global_step
        for batch_xs, batch_ys in valid_loader:
            batch_xs = batch_xs.to(device)
            batch_ys = batch_ys.to(device)
            pred = model(batch_xs)
            loss = loss_fn(pred, batch_ys)
            logger.add_scalar("validation loss", loss, global_step=val_step)
            val_step += 1

    return model


model = train()
torch.save(model, "bigger_anomaly.pt")
# model = torch.load("bigger_anomaly.pt")
model.eval()
full_loader = DataLoader(anomaly_dataset, batch_size=8, shuffle=False)
preds = []
for batch, _ in full_loader:
    batch = batch.to(device)
    batch_pred = model(batch)
    for pred in batch_pred:
        preds.append(pred[-1].item())
preds = np.array(preds)
np.savez("bigger_anomaly_preds.npz", preds=preds)
