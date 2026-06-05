import torch
import json
import torch.nn as nn
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datafile    = "improved-system\\genericWeightInferrer\\hourly\\adelaide_weather_testset_hourly.csv"
weightsfile = "improved-system\\genericWeightInferrer\\hourly\\longerweights_hourly.pt"
normstatsfile = "improved-system\\genericWeightInferrer\\hourly\\norm_stats_hourly.json"

df = pd.read_csv(datafile)

with open(normstatsfile) as f:
    norm_stats = json.load(f)

def normalize(v, col):
    mn = norm_stats[col]["min"]
    mx = norm_stats[col]["max"]
    return (v - mn) / (mx - mn) if mx != mn else 0.0


# MUST match training architecture exactly
model = nn.Sequential(
    nn.Linear(9, 16),  nn.ReLU(),
    nn.Linear(16, 12), nn.ReLU(),
    nn.Linear(12, 8),  nn.ReLU(),
    nn.Linear(8, 4),   nn.ReLU(),
    nn.Linear(4, 1),   nn.Sigmoid()
).to(device)

# Load trained weights
model.load_state_dict(torch.load(weightsfile, map_location=device))
model.eval()


success = 0
fail = 0
err_history = []

with torch.no_grad():
    for row in df.itertuples():

        x = np.array([
            normalize(row.temperature,   "temperature"),
            normalize(row.humidity,      "humidity"),
            normalize(row.pressure,      "pressure"),
            normalize(row.wind_speed,    "wind_speed"),
            row.season_sin,
            row.season_cos,
            row.hour_sin,
            row.hour_cos,
            normalize(row.rain_lasthour, "rain_lasthour")
        ], dtype=np.float32)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

        yhat = model(x).item()
        y = 1 if yhat >= 0.5 else 0

        rain = int(row.rain)

        err_history.append(rain - y)

        if rain == y:
            success += 1
        else:
            fail += 1

        print(f"Pred: {'rain' if y else 'no rain'} ({yhat:.3f}) | True: {'rain' if rain else 'no rain'}")


total = success + fail
print(f"\nAccuracy: {100 * success / total:.4f}%")
print("FP:", err_history.count(-1))
print("TN/TP:", err_history.count(0))
print("FN:", err_history.count(1))