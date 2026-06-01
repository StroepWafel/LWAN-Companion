import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Filepaths
datafile = "improved-system\\genericWeightInferrer\\adelaide_weather_testset.csv"
weightsfile = "improved-system\\genericWeightInferrer\\longerweights.json"

# Import data
df = pd.read_csv(datafile)

# Load norm stats
with open("improved-system\\genericWeightInferrer\\norm_stats.json") as f:
    norm_stats = json.load(f)

# Load weights and build matrices automatically from saved architecture
try:
    with open(weightsfile) as file:
        saved = json.load(file)
except Exception as e:
    print(f"Failed to load weights from file: {e}")
    exit()

weights = [np.array(layer["weights"]) for layer in saved["layers"]]
biases  = [np.array(layer["bias"]) if isinstance(layer["bias"], list)
           else float(layer["bias"]) for layer in saved["layers"]]

print(f"Loaded network architecture: {saved['architecture']}")

# Statistics
failcount = 0
successcount = 0

# Tracking over time
success_history = []
fail_history = []

predict_history = []
expect_history = []
err_history = []

# Normalize a single entry before inference
def normalize(value, col):
    mn = norm_stats[col]["min"]
    mx = norm_stats[col]["max"]
    if mx != mn:
        return (value - mn) / (mx - mn)
    return 0.0

# Predict weather
for row in df.itertuples():
    # Inputs (6 features)
    x = np.array([
        normalize(row.temperature, "temperature"),
        normalize(row.humidity, "humidity"),
        normalize(row.pressure, "pressure"),
        normalize(row.wind_speed, "wind_speed"),
        row.season_sin,
        row.season_cos,
        normalize(row.rain_yesterday, "rain_yesterday")
        ])
    rain = 1 if row.rain == 1 else 0

    # --- Forward pass ---
    a = x
    for idx, (W, b) in enumerate(zip(weights, biases)):
        z = W @ a + b
        if idx < len(weights) - 1:
            a = np.maximum(0, z)          # ReLU for hidden layers
        else:
            a = 1 / (1 + np.exp(-z))      # sigmoid only on output
    yhat = float(np.squeeze(a))
    y = 1 if yhat >= 0.4 else 0


    predict_history.append(y)
    expect_history.append(rain)
    err_history.append(int(rain) - int(y))

    # Increment counter
    if rain == y:
        successcount += 1
    else:
        failcount += 1

    success_history.append(successcount)
    fail_history.append(failcount)
    
    # Log
    print(f"Predicted: {'   rain' if y else 'no rain'} ({yhat:.2f}), Expected: {'   rain' if rain else 'no rain'}")

print(f"Succeeded {successcount} times, Failed {failcount} times. Accuracy: {100*successcount/(successcount+failcount):.4f}")
print(f"{err_history.count(-1)} instances of predicting rain when not expected")
print(f"{err_history.count(0)} instances of predicting what was expected")
print(f"{err_history.count(1)} instances of predicting no rain when rain expected")

# Plot for analytics
# plt.figure(figsize=(10, 5))
# plt.plot(success_history, label="Successes", color="green")
# plt.plot(fail_history, label="Failures", color="red")
# plt.xlabel("Row")
# plt.ylabel("Cumulative count")
# plt.title("MLP Inferring: Successes vs Failures Over Time")
# plt.legend()
# plt.tight_layout()
# plt.show()
# plt.figure(figsize=(30, 5))
# plt.plot(predict_history, label="Predicted", color="green")
# plt.plot(expect_history, label="Expected", color="blue")
# plt.xlabel("Row")
# plt.ylabel("1 = Rain")
# plt.title("MLP Inferring: Expected vs Predicted Over Time")
# plt.legend()
# plt.tight_layout()
# plt.show()
# plt.figure(figsize=(30, 5))
# plt.plot(err_history, label="Difference", color="green")
# plt.xlabel("Row")
# plt.ylabel("Expected - Prediction")
# plt.title("MLP Inferring: Difference in prediction Over Time (0 is good, -1 means predicted rain when no rain expected, 1 means vice versa)")
# plt.legend()
# plt.tight_layout()
# plt.show()