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

# TODO: Try increasing the sequence length and see what happens
points = 52077
train_size = int(0.8 * points)
seq_len = 512
logdir = "logs"
modeldir = "models"
predsdir = "preds"
predictive = False


class TempAnomalyDataset(Dataset):
    def __init__(self, file_path: str, seq_len: int, is_predictive: bool):
        self.data = pd.read_json(file_path)
        self.data = self.data.to_numpy()
        self.data = torch.tensor(self.data[:, 4])
        self.seq_len = seq_len
        self.is_predictive = is_predictive

    def __len__(self):
        if self.is_predictive:
            return len(self.data) - 2 * self.seq_len - 1
        else:
            return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        # Out of bound indices
        if idx > len(self.data):
            idx = len(self.data) - 1
        elif idx < self.seq_len:
            idx = self.seq_len
        if self.is_predictive:
            return self.data[idx - self.seq_len : idx], self.data[
                idx : idx + self.seq_len
            ]
        else:
            return self.data[idx - self.seq_len : idx], self.data[
                idx - self.seq_len + 1 : idx + 1
            ]


anomaly_dataset = TempAnomalyDataset(
    "data/berkeley/data.json", seq_len=seq_len, is_predictive=predictive
)
train_data, valid_data = torch.utils.data.random_split(
    anomaly_dataset, [train_size, len(anomaly_dataset) - train_size]
)

df = pd.read_json("data/berkeley/data.json")
time_series = df[["anomaly"]].values.astype("float32")
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
        self.classifier = nn.Conv1d(c, 1, 1, dtype=float)

    def forward(self, x) -> torch.Tensor:
        # Adding our features so convolutions don't get mad
        x = torch.unsqueeze(x, 1)
        # Feeding our data into our network
        x = self.network(x)
        # Classifying our features
        x = self.classifier(x)
        # Removing size 1 dimension where our features were
        return torch.squeeze(x, 1)


def train(
    lr: float, epochs: int, weight_decay: float, layers: list[int], batch_size: int
) -> TempAnomalyNetwork:
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    logger = tb.SummaryWriter(
        logdir
        + "/"
        + "lr-"
        + str(lr)
        + "-wght-"
        + str(weight_decay)
        + "-arch-"
        + str(layers[-1])
        + "-ep-"
        + str(epochs)
        + "-seqlen-"
        + str(seq_len)
        + "-pred-"
        + str(predictive)
    )

    model = TempAnomalyNetwork(layers=layers).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    global_step = 0
    val_step = 0

    for i in range(epochs):
        model.train()
        for batch_xs, batch_ys in train_loader:
            batch_xs = batch_xs.to(device)
            batch_ys = batch_ys.to(device)
            pred = model(batch_xs)
            loss = loss_fn(pred, batch_ys)
            logger.add_scalar("train_loss", loss.item(), global_step=global_step)
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1
        model.eval()
        losses = []
        for batch_xs, batch_ys in valid_loader:
            batch_xs = batch_xs.to(device)
            batch_ys = batch_ys.to(device)
            pred = model(batch_xs)
            loss = loss_fn(pred, batch_ys)
            logger.add_scalar("validation loss", loss.item(), global_step=val_step)
            losses.append(loss.item())
            val_step += 1
        print("Epoch: ", i, "Average validation loss: ", np.mean(losses))

    return model


# TODO: Try a bigger model?
weight_decay = 5e-4
lr = 5e-3
layers = [8, 16, 32, 64]
epochs = 3
batch_size = 8


model = train(
    epochs=epochs,
    weight_decay=weight_decay,
    lr=lr,
    layers=layers,
    batch_size=batch_size,
)
torch.save(
    model,
    modeldir
    + "/"
    + "lr-"
    + str(lr)
    + "-wght-"
    + str(weight_decay)
    + "-arch-"
    + str(layers[-1])
    + "-ep-"
    + str(epochs)
    + "-seqlen-"
    + str(seq_len)
    + "-pred-"
    + str(predictive),
)

# All of the below is modelling in the predictive case
model.eval()
if predictive:
    partitions = points // seq_len
    new_points = partitions * seq_len
    partitioned_dataset = anomaly_dataset.data[0:new_points]
    partitioned_dataset = torch.split(partitioned_dataset, seq_len)
    preds = np.array([])
    for part in partitioned_dataset:
        part = part.unsqueeze(dim=0)
        batch_preds = model(part).squeeze()
        batch_preds = batch_preds.detach().numpy()
        preds = np.concat((preds, batch_preds))
    plt.plot(preds, c="r")

    # Feeding the model back its own predictions
    model.eval()
    data = partitioned_dataset[-1].unsqueeze(dim=1)
    future_preds = np.ones_like(preds) * np.nan
    i = 0
    while i < 20:
        batch_pred = model(data)
        data = batch_pred
        batch_pred = batch_pred.detach().numpy().squeeze()
        future_preds = np.concat((future_preds, batch_pred))
        i += 1
    # plt.plot(future_preds, c="y")
    np.savez(
        "future_preds"
        + "-lr-"
        + str(lr)
        + "-wght-"
        + str(weight_decay)
        + "-arch-"
        + str(layers[-1])
        + "-ep-"
        + str(epochs)
        + "-seqlen-"
        + str(seq_len)
        + "-pred-"
        + str(predictive),
        future_preds,
    )
    # plt.show()

# All of the below is modelling the non-predictive case
# TODO: This always seems to "worm" where the predictions just tend to a value near 1
else:
    full_loader = DataLoader(anomaly_dataset, batch_size=8, shuffle=False)
    preds = []
    for batch, _ in full_loader:
        batch = batch.to(device)
        batch_pred = model(batch)
        for pred in batch_pred:
            preds.append(pred[-1].item())
    preds = np.array(preds)
    preds = np.expand_dims(preds, axis=1)
    plt.plot(preds, c="r")

    model.eval()
    data = torch.split(anomaly_dataset.data, seq_len)[-1].unsqueeze(dim=1)
    future_preds = np.ones_like(preds) * np.nan
    i = 0
    while i < 3000:
        batch_pred = model(data)
        data = torch.concat((data[0 : seq_len - 1], batch_pred.unsqueeze()[-1]), dim=0)
        batch_pred = batch_pred.detach().numpy()
        future_preds = np.append(future_preds, batch_pred[-1])
        i += 1
    np.savez("future_preds.npz", future_preds)
    # plt.plot(future_preds, c="y")
    np.savez(
        "future_preds"
        + "-lr-"
        + str(lr)
        + "-wght-"
        + str(weight_decay)
        + "-arch-"
        + str(layers[-1])
        + "-ep-"
        + str(epochs)
        + "-seqlen-"
        + str(seq_len)
        + "-pred-"
        + str(predictive),
        future_preds,
    )
    # plt.show()
