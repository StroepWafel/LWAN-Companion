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

# Epoch count
epochs = 100


# # Statistics
# global totalfailcount 
# totalfailcount = 0
# global totalsuccesscount
# totalsuccesscount = 0
# total_success_history = []
# total_fail_history = []
    

# Train on every row of data
for epoch in range(epochs):
    successcount = 0
    failcount = 0
    success_history = []
    fail_history = []

    for row in df.itertuples():
        # Inputs
        x = np.array([row.temperature, row.humidity, row.pressure, row.wind_speed])
        # Expected output
        rain = row.rain

        # Forward Pass Mathematics
        z1 = W1 @ x + b1                    # Matrix Vector for weighted sum
        a1 = 1 / (1 + np.exp(-z1))          #  
        z2 = W2 @ a1 + b2                   # 
        yhat = 1 / (1 + np.exp(-z2))        # 
        y = 1 if yhat >= 0.5 else 0         # 

        # Backward Pass Mathematics
        d2 = yhat - rain #type: ignore      # 
        d1 = (W2.T @ d2) * a1 * (1 - a1)    # 


        # Weight Updates
        W2 -= n * np.outer(d2, a1)          # 
        b2 -= n * d2.item()                 # 
        W1 -= n * np.outer(d1, x)           # 
        b1 -= n * d1                        # 

        y = 1 if (z >= 0) else 0

        w = w + n * (rain - y) * x #type: ignore
        b = b + n * (rain - y) #type: ignore
        if (rain == y):
            successcount += 1
        else:
            failcount += 1

        success_history.append(successcount)
        fail_history.append(failcount)

    print(f"EPOCH {epoch+1}/{epochs} --- Accuracy: {successcount/(successcount+failcount):.4f}%")

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