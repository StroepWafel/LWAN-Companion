import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Filepaths
datafile = "improved-system\\adelaide_weather_testset.csv"
weightsfile = "improved-system\\weights.json"
weighting = {}

# Import data
df = pd.read_csv(datafile)
try:
    with open(weightsfile) as file:
        weighting = json.load(file)
except Exception as e:
    print(f"Failed to load weights from file: {e}")

with open("improved-system\\genericWeightInferrer\\norm_stats.json") as f:
    norm_stats = json.load(f)

# Normalize a single entry before inference
def normalize(value, col):
    mn = norm_stats[col]["min"]
    mx = norm_stats[col]["max"]
    if mx != mn:
        return (value - mn) / (mx - mn)
    return 0.0


W1 = np.array(weighting["W1"])
b1 = np.array(weighting["b1"])
W2 = np.array(weighting["W2"])
b2 = np.array(weighting["b2"])
W3 = np.array(weighting["W3"])
b3 = weighting["b3"]

# Statistics
failcount = 0
successcount = 0

# Tracking over time
success_history = []
fail_history = []

# Predict weather
for row in df.itertuples():
    # Inputs (6 features)
    x = np.array([
        normalize(row.temperature, "temperature"),
        normalize(row.humidity, "humidity"),
        normalize(row.pressure, "pressure"),
        normalize(row.wind_speed, "wind_speed"),
        row.season_sin,
        row.season_cos
        ])
    # Expected output
    rain = row.rain

    # Forward Pass Mathematics
    z1 = W1 @ x + b1                    # Matrix Vector for weighted sum
    a1 = 1 / (1 + np.exp(-z1))          # Sigmoid activation for calculated vector
    z2 = W2 @ a1 + b2                   # Matrix Vector for weighted sum - 2nd layer
    a2 = 1 / (1 + np.exp(-z2))          # Sigmoid activation for second layer calculated vector
    z3 = W3 @ a2 + b3                   # Matrix Vector for weighted sum - 3rd layer (output)
    yhat = 1 / (1 + np.exp(-z3))        # Sigmoid activation for calculated vector - 2nd layer (output)
    y = 1 if yhat >= 0.5 else 0         # Threshold for output/prediction (rain or no rain)

    # Increment counter
    if rain == y:
        successcount += 1
    else:
        failcount += 1

    success_history.append(successcount)
    fail_history.append(failcount)

    # Log
    print(f"Predicted: {'   rain' if y else 'no rain'}, Expected: {'   rain' if rain else 'no rain'}")

print(f"Succeeded {successcount} times, Failed {failcount} times. Accuracy: {100*successcount/(successcount+failcount):.4f}")

# Plot for analytics
plt.figure(figsize=(10, 5))
plt.plot(success_history, label="Successes", color="green")
plt.plot(fail_history, label="Failures", color="red")
plt.xlabel("Row")
plt.ylabel("Cumulative count")
plt.title("MLP Inferring: Successes vs Failures Over Time")
plt.legend()
plt.tight_layout()
plt.show()
