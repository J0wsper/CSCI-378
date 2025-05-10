import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json("data/berkeley/data.json")
time_series = df[["anomaly"]].values.astype("float32")
plt.plot(time_series, c="b")

preddir = "preds/"

lr = 5e-4
weight = 0.0
arch = 128
epochs = 15
seqlen = 512
pred = True

preds = np.load(
    preddir
    + "lr-"
    + str(lr)
    + "-wght-"
    + str(weight)
    + "-arch-"
    + str(arch)
    + "-ep-"
    + str(epochs)
    + "-seqlen-"
    + str(seqlen)
    + "-pred-"
    + str(pred)
    + ".npz"
)
padding = np.ones((seqlen,)) * np.nan
preds = np.concat((padding, preds))
plt.plot(preds, c="r")

future_preds = np.load(
    preddir
    + "future_preds"
    + "-lr-"
    + str(lr)
    + "-wght-"
    + str(weight)
    + "-arch-"
    + str(arch)
    + "-ep-"
    + str(epochs)
    + "-seqlen-"
    + str(seqlen)
    + "-pred-"
    + str(pred)
    + ".npz"
)
future_preds = future_preds["arr_0"]
plt.plot(future_preds, c="y")
plt.show()
