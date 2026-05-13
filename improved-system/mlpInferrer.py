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

w = np.array(weighting["weights"])
b = weighting["bias"]
# Statistics
global failcount 
failcount = 0
global successcount
successcount = 0

# Tracking over time
success_history = []
fail_history = []

# Predict weather
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

    # Successful?
    y = 1 if (z >= 0) else 0

    # Increment counter
    if (rain == y):
        successcount += 1
    else:
        failcount += 1

    success_history.append(successcount)
    fail_history.append(failcount)

    # Log
    print(f"Predicted: {'   rain' if y else 'no rain'}, Expected: {'   rain' if rain else 'no rain'}.")


print(f"Succeeded {successcount} times, Failed {failcount} times. Accuracy: {successcount/(successcount+failcount)}")


# Plot for analytics
plt.figure(figsize=(10, 5))
plt.plot(success_history, label="Successes", color="green")
plt.plot(fail_history, label="Failures", color="red")
plt.xlabel("Row")
plt.ylabel("Cumulative count")
plt.title("Perceptron Inferring: Successes vs Failures Over Time")
plt.legend()
plt.tight_layout()
plt.show()