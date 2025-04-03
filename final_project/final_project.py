# import torchvision.transforms.v2 as transforms
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pinnstorch
# import torchvision
import pandas as pd
# torchvision.disable_beta_transforms_warning()

# Number of data points in our set

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"


class TempAnomalyDataset(Dataset):
    def __init__(self, file_path):
        # This data set has shape (51830, 5)
        # I removed leap days for ease of putting the data into numpy arrays
        self.anomalies = pd.read_json(file_path)
        # 142 years between 1880 and 2021, each of which has 365 days
        self.anomalies = np.asarray(
            self.anomalies, dtype=float).reshape(142, 365, 5)
        # Selecting out the anomalies because the other data is implicit in the
        # structure of the array now.
        self.anomalies = self.anomalies[:, :, 4]

        def __len__(self):
            return len(self.anomalies)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            return self.anomalies[idx]


anomaly_dataset = TempAnomalyDataset("data/berkeley/anomalies.json")
print(anomaly_dataset)
