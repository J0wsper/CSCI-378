import torch
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt
import pinnstorch

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
