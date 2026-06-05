import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

datafile    = "improved-system\\genericWeightInferrer\\hourly\\adelaide_weather_dataset_hourly.csv"
weightsfile = "improved-system\\genericWeightInferrer\\hourly\\longerweights_hourly.pt"

df = pd.read_csv(datafile)

FEATURES = ["temperature", "humidity", "pressure", "wind_speed",
            "season_sin", "season_cos", "hour_sin", "hour_cos", "rain_lasthour"]

# Load entire dataset to GPU at once
X = torch.tensor(df[FEATURES].values, dtype=torch.float32).to(device)
y = torch.tensor(df["rain"].values,   dtype=torch.float32).to(device)
print(f"Dataset on GPU: {X.shape} — {X.nbytes / 1e6:.1f} MB")

# Model
model = nn.Sequential(
    nn.Linear(9, 16),  nn.ReLU(),
    nn.Linear(16, 12), nn.ReLU(),
    nn.Linear(12, 8),  nn.ReLU(),
    nn.Linear(8, 4),   nn.ReLU(),
    nn.Linear(4, 1),   nn.Sigmoid()
).to(device)

# Class weights
rain_rate     = df["rain"].mean()
weight_rain   = float(1 - rain_rate)
weight_norain = float(rain_rate)
print(f"Rain rate: {rain_rate:.4f} | weight_rain: {weight_rain:.4f} | weight_norain: {weight_norain:.4f}")

sample_weights = torch.where(y == 1,
    torch.tensor(weight_rain,   device=device),
    torch.tensor(weight_norain, device=device))

criterion = nn.BCELoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Early stopping
min_epochs             = 1400
patience               = 200
best_accuracy          = 0.0
epochs_without_improvement = 0
best_state             = None

for epoch in range(1, 10001):
    model.train()

    # Full dataset forward + backward in one shot
    optimizer.zero_grad()
    yhat = model(X).squeeze()
    loss = (criterion(yhat, y) * sample_weights).mean()
    loss.backward()
    optimizer.step()

    # Accuracy (no_grad for speed)
    with torch.no_grad():
        preds    = (yhat >= 0.5).float()
        accuracy = (preds == y).float().mean().item()

    print(f"EPOCH {epoch} --- Accuracy: {accuracy * 100:.4f}% | Loss: {loss.item():.6f}")

    if epoch >= min_epochs:
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} — best accuracy: {best_accuracy * 100:.4f}%")
                break

# Restore and save best model
model.load_state_dict(best_state)
torch.save(best_state, weightsfile)
print(f"Saved best weights to {weightsfile}")