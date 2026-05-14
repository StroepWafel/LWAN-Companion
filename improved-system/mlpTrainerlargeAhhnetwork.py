import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datafile = "improved-system\\adelaide_weather_dataset.csv"
weightsfile = "improved-system\\longerweights.json"

# Import data
df = pd.read_csv(datafile)

# --- Network architecture ---
# Define hidden layer sizes, input and output are automatic
N_INPUTS = 6
N_OUTPUTS = 1
HIDDEN_LAYERS = [8, 10, 5, 6]  # change this to whatever you want

# Build full layer size list: [n_inputs, h1, h2, ..., n_outputs]
layer_sizes = [N_INPUTS] + HIDDEN_LAYERS + [N_OUTPUTS]

# --- Xavier uniform initialisation ---
# For each pair of adjacent layers, create a weight matrix and bias vector
weights = []
biases = []
for i in range(len(layer_sizes) - 1):
    n_in = layer_sizes[i]
    n_out = layer_sizes[i + 1]
    bound = np.sqrt(6 / (n_in + n_out))
    W = np.random.uniform(-bound, bound, (n_out, n_in))
    b = np.zeros(n_out) if i < len(layer_sizes) - 2 else 0.0  # scalar bias for output layer
    weights.append(W)
    biases.append(b)

# Learning rate
n = 0.01

# Early stopping settings
min_epochs = 100
patience = 10
best_accuracy = 0.0
epochs_without_improvement = 0
epoch = 0

# Statistics
success_history = []
fail_history = []
accuracy_history = []

# --- Training loop ---
while True:
    epoch += 1
    successcount = 0
    failcount = 0

    for row in df.itertuples():
        # Inputs
        x = np.array([row.temperature, row.humidity, row.pressure, row.wind_speed, row.season_sin, row.season_cos])
        rain = row.rain

        # --- Forward pass ---
        # Pass input through each layer, storing activations
        activations = [x]
        a = x
        for W, b in zip(weights, biases):
            z = W @ a + b
            a = 1 / (1 + np.exp(-z))   # sigmoid for all layers including output
            activations.append(a)

        yhat = activations[-1].item()   # squeeze to Python scalar
        y = 1 if yhat >= 0.5 else 0    # threshold to binary prediction

        # --- Backward pass ---
        # Compute deltas from output layer back to first hidden layer
        deltas = [np.zeros(W.shape[0]) for W in weights]
        deltas[-1] = np.array([yhat - rain])  # output delta, shape (1,)

        for i in range(len(weights) - 2, -1, -1):
            a_i = activations[i + 1]    # activation of layer i
            deltas[i] = (weights[i + 1].T @ deltas[i + 1]) * a_i * (1 - a_i)

        # --- Weight updates ---
        for i in range(len(weights)):
            weights[i] -= n * np.outer(deltas[i], activations[i])
            if i == len(weights) - 1:
                biases[i] -= n * float(deltas[i].item())  # scalar output bias
            else:
                biases[i] -= n * deltas[i]              # vector hidden bias

        if rain == y:
            successcount += 1
        else:
            failcount += 1

    success_history.append(successcount)
    fail_history.append(failcount)
    accuracy = successcount / (successcount + failcount)
    accuracy_history.append(accuracy)
    print(f"EPOCH {epoch} --- Accuracy: {accuracy * 100:.4f}%")

    if epoch >= min_epochs:
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} — no improvement for {patience} epochs")
                break

# --- Save weights ---
print("Saving final weights to file...")
try:
    export = {
        "architecture": layer_sizes,
        "layers": []
    }
    for i, (W, b) in enumerate(zip(weights, biases)):
        n_out, n_in = W.shape
        layer_entry = {
            "layer": i + 1,
            "order": f"{n_out}x{n_in}",
            "weights": W.tolist(),
            "bias": b.tolist() if isinstance(b, np.ndarray) else float(b)
        }
        export["layers"].append(layer_entry)

    with open(weightsfile, "w") as file:
        json.dump(export, file, indent=2)
        print(f"Successfully saved weights to `{file.name}`")
except Exception as e:
    print(f"Failed to write weights to file: {e}")

# --- Plots ---
plt.figure(figsize=(10, 5))
plt.plot(success_history, label="Successes", color="green")
plt.plot(fail_history, label="Failures", color="red")
plt.xlabel("Epoch")
plt.ylabel("Count per epoch")
plt.title("MLP Training: Successes vs Failures per Epoch")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(accuracy_history, label="Accuracy", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MLP Training: Accuracy Over Time")
plt.legend()
plt.tight_layout()
plt.show()