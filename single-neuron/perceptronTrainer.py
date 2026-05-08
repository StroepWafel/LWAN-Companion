import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datafile = "single-neuron\\adelaide_weather_dataset.csv"
weightsfile = "single-neuron\\weights.json"

# Import data
df = pd.read_csv(datafile)

# Initialise weights
b = 0
global w
w = np.array([0, 0, 0, 0])

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