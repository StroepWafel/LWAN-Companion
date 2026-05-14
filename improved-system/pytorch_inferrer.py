import torch
import torch.nn as nn
import pandas as pd
import json
import matplotlib.pyplot as plt

DATAFILE    = "improved-system\\adelaide_weather_testset.csv"
MODELFILE   = "improved-system\\best_model.pt"
GENOMEFILE  = "improved-system\\best_genome.json"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load genome to reconstruct architecture ---
with open(GENOMEFILE) as f:
    genome = json.load(f)

print(f"Loaded architecture: {genome['hidden_layers']}, dropout: {genome['dropout']}")

# --- Rebuild network (must match trainer exactly) ---
def build_network(hidden_layers: list, dropout: float = 0.0) -> nn.Sequential:
    layer_sizes = [6] + hidden_layers + [1]
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Sigmoid())
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

model = build_network(genome["hidden_layers"], genome["dropout"])
model.load_state_dict(torch.load(MODELFILE, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- Load test data ---
df = pd.read_csv(DATAFILE)
features = ["temperature", "humidity", "pressure", "wind_speed", "season_sin", "season_cos"]
X = torch.tensor(df[features].values, dtype=torch.float32).to(DEVICE)
y = df["rain"].values

# --- Infer ---
successcount = 0
failcount = 0
success_history = []
fail_history = []

with torch.no_grad():
    for i, row in enumerate(X):
        yhat = model(row.unsqueeze(0)).item()
        pred = 1 if yhat >= 0.5 else 0
        expected = int(y[i])

        if pred == expected:
            successcount += 1
        else:
            failcount += 1

        success_history.append(successcount)
        fail_history.append(failcount)

        print(f"Predicted: {'   rain' if pred else 'no rain'} ({yhat:.2f}), "
              f"Expected: {'   rain' if expected else 'no rain'}")

print(f"\nSucceeded {successcount} times, Failed {failcount} times. "
      f"Accuracy: {successcount/(successcount+failcount):.4f}")

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(success_history, label="Successes", color="green")
plt.plot(fail_history, label="Failures", color="red")
plt.xlabel("Row")
plt.ylabel("Cumulative count")
plt.title("Evolutionary MLP Inferring: Successes vs Failures")
plt.legend()
plt.tight_layout()
plt.show()