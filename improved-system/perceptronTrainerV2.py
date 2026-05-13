import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datafile = "improved-system\\adelaide_weather_dataset.csv"
weightsfile = "improved-system\\weights.json"

# Import data
df = pd.read_csv(datafile)

# Xavier bounds for each layer
# W1: 4 inputs, 8 outputs -> n_in=4, n_out=8 -> sqrt(6/12)
# W2: 8 inputs, 1 output  -> n_in=8, n_out=1 -> sqrt(6/9)
W1 = np.random.uniform(-np.sqrt(6/12), np.sqrt(6/12), (8, 4))
b1 = np.zeros(8)
W2 = np.random.uniform(-np.sqrt(6/9), np.sqrt(6/9), (8, 4))
b2 = 0.0

# Learning rate
n = 0.01


# Statistics
global failcount 
failcount = 0
global successcount
successcount = 0

# Tracking over time
success_history = []
fail_history = []

# Train on every row of data
for row in df.itertuples():
    # Inputs
    temperature = row.temperature
    humidity = row.humidity
    pressure = row.pressure
    wind_speed = row.wind_speed
    # Expected output
    rain = row.rain

    # Create vector x
    x = np.array([temperature, humidity, pressure, wind_speed])
    # Create output
    z = x.dot(w) + b

    
    y = 1 if (z >= 0) else 0

    w = w + n * (rain - y) * x #type: ignore
    b = b + n * (rain - y) #type: ignore
    if (rain == y):
        successcount += 1
    else:
        failcount += 1

    success_history.append(successcount)
    fail_history.append(failcount)

    print(f"Predicted {'rain' if y else 'no rain'}, Expected {'rain' if rain else 'no rain'}. New weights: {w.tolist()}, New bias: {b}")


print(f"Succeeded {successcount} times, Failed {failcount} times. Accuracy: {successcount/(successcount+failcount)}")

print("Saving final weights to file...")
try:
    export = {
        "weights": w.tolist(),
        "bias": b
    }
    with open(weightsfile, "w") as file:
        json.dump(export, file)
        print(f"Successfully saved weights {w.tolist()} and bias {b} to `{file.name}`")
except Exception as e:
    print(f"Failed to write weights to file: {e}")



# Plot for analytics
plt.figure(figsize=(10, 5))
plt.plot(success_history, label="Successes", color="green")
plt.plot(fail_history, label="Failures", color="red")
plt.xlabel("Row")
plt.ylabel("Cumulative count")
plt.title("Perceptron Training: Successes vs Failures Over Time")
plt.legend()
plt.tight_layout()
plt.show()