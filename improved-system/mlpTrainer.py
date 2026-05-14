import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datafile = "improved-system\\adelaide_weather_dataset.csv"
weightsfile = "improved-system\\weights.json"

# Import data
df = pd.read_csv(datafile)

# Xavier bounds for each layer
# W1: 6 inputs, 8 outputs  -> n_in=6, n_out=8 -> sqrt(6/14)
# W2: 8 inputs, 8 outputs  -> n_in=8, n_out=8 -> sqrt(6/16)
# W3: 8 inputs, 1 output   -> n_in=8, n_out=1 -> sqrt(6/9)
W1 = np.random.uniform(-np.sqrt(6/14), np.sqrt(6/14), (8, 6))
b1 = np.zeros(8)
W2 = np.random.uniform(-np.sqrt(6/16), np.sqrt(6/16), (8, 8))
b2 = np.zeros(8)
W3 = np.random.uniform(-np.sqrt(6/9), np.sqrt(6/9), (1, 8))
b3 = 0.0

# Learning rate
n = 0.01

# Epoch count
epochs = 9999


# Statistics
global totalfailcount 
totalfailcount = 0
global totalsuccesscount
totalsuccesscount = 0
global success_history
success_history = []
global fail_history
fail_history = []
global ratio_history
accuracy_history = []
    
# Early stopping settings
min_epochs = 100        # don't stop before this
patience = 10          # stop after this many epochs with no improvement
best_accuracy = 0.0
epochs_without_improvement = 0
epoch = 0


# Train on every row of data repeatedly until stagnation
while True:
    epoch += 1
    successcount = 0
    failcount = 0

    for row in df.itertuples():
        # Inputs
        x = np.array([row.temperature, row.humidity, row.pressure, row.wind_speed, row.season_sin, row.season_cos])
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

        # Backward Pass Mathematics
        d3 = yhat - rain #type: ignore      # Output delta (y^ - d)
        d2 = (W3.T @ d3) * a2 * (1 - a2)    # Hidden layer 2 delta calculation (δ^((1) )  =(W^((2) ) )^T δ^((2) )⊙a^((1) )⊙(1-a^((1) ) ))
        d1 = (W2.T @ d2) * a1 * (1 - a1)    # Hidden layer 1 delta calculation (δ^((1) )  =(W^((2) ) )^T δ^((2) )⊙a^((1) )⊙(1-a^((1) ) ))

        # Weight Updates
        W3 -= n * np.outer(d3, a2)          # Update 3rd (output) layer weights
        b3 -= n * d3.item()                 # Update 3rd (output) layer bias
        W2 -= n * np.outer(d2, a1)          # Update 2nd layer weights
        b2 -= n * d2                        # Update 2nd layer bias
        W1 -= n * np.outer(d1, x)           # Update 1st layer weights
        b1 -= n * d1                        # Update 1st layer bias

        if (rain == y):
            successcount += 1
        else:
            failcount += 1

    success_history.append(successcount)
    fail_history.append(failcount)

    totalfailcount += successcount
    totalsuccesscount += failcount
    accuracy = successcount/(successcount+failcount)
    accuracy_history.append(accuracy)
    print(f"EPOCH {epoch+1}/{epochs} --- Accuracy: {(accuracy*100):.4f}%")

    if epoch >= min_epochs:
        if accuracy != best_accuracy:
            best_accuracy = accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} — no change for {patience} epochs")
                break


print("Saving final weights to file...")
try:
    export = {
        "W1": W1.tolist(),
        "b1": b1.tolist(),
        "W2": W2.tolist(),
        "b2": b2.tolist(),
        "W3": W3.tolist(),
        "b3": float(b3.real)
    }
    with open(weightsfile, "w") as file:
        json.dump(export, file, indent=2)
        print(f"Successfully saved weights to `{file.name}`")
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

plt.figure(figsize=(10, 5))
plt.plot(accuracy_history, label="Successes", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Success Percent")
plt.title("Perceptron Training: Success Percent Over Time")
plt.legend()
plt.tight_layout()
plt.show()