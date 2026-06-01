import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datafile = "improved-system\\genericWeightInferrer\\adelaide_weather_dataset.csv"
weightsfile = "improved-system\\genericWeightInferrer\\longerweights.json"

# Import data
df = pd.read_csv(datafile)

# --- Network architecture ---
# Define hidden layer sizes, input and output are automatic
N_INPUTS = 7
N_OUTPUTS = 1
HIDDEN_LAYERS = [16, 12, 8, 4]  # change this to whatever

# How large should our batches be?
BATCH_SIZE = 128

# Build full layer size list: [n_inputs, h1, h2, ..., n_outputs]
layer_sizes = [N_INPUTS] + HIDDEN_LAYERS + [N_OUTPUTS]

# Calculate class weight
rain_rate = df['rain'].mean()
weight_rain = 0.65
weight_norain = 0.35
# weight_rain = (1 - rain_rate)      # weight for rain=1 samples
# weight_norain = rain_rate          # weight for rain=0 samples
print(f"weight rain: {weight_rain}, weight no rain: {weight_norain}")
# --- Xavier uniform initialisation ---
# For each pair of adjacent layers, create a weight matrix and bias vector
weights = []
biases = []
best_weights = [W.copy() for W in weights]
best_biases = [b.copy() if isinstance(b, np.ndarray) else b for b in biases]
for i in range(len(layer_sizes) - 1):
    n_in = layer_sizes[i]
    n_out = layer_sizes[i + 1]
    bound = np.sqrt(6 / (n_in + n_out))
    W = np.random.uniform(-bound, bound, (n_out, n_in))
    b = np.zeros(n_out) if i < len(layer_sizes) - 2 else 0.0  # scalar bias for output layer
    weights.append(W)
    biases.append(b)

# Learning rate
n = 0.003

# Early stopping settings
min_epochs = 250
patience = 100
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

    df_shuffled = df.sample(frac=1).reset_index(drop=True)

    for batch_start in range(0, len(df_shuffled), BATCH_SIZE):
        batch = df_shuffled.iloc[batch_start : batch_start + BATCH_SIZE]

        grad_W = [np.zeros_like(W) for W in weights]
        grad_b = [np.zeros_like(b) for b in biases]

        for row in batch.itertuples():
            x = np.array([row.temperature, row.humidity, row.pressure, row.wind_speed,
                           row.season_sin, row.season_cos, row.rain_yesterday])
            rain = row.rain

            # --- Forward pass ---
            activations = [x]
            a = x
            for idx, (W, b) in enumerate(zip(weights, biases)):
                z = W @ a + b
                if idx < len(weights) - 1:
                    a = np.maximum(0, z)          # ReLU for hidden layers
                else:
                    a = 1 / (1 + np.exp(-z))      # sigmoid only on output
                activations.append(a)

            
            yhat = activations[-1].item()
            y = 1 if yhat >= 0.5 else 0

            # --- Backward pass ---
            deltas = [np.zeros(W.shape[0]) for W in weights]
            w = weight_rain if rain == 1 else weight_norain
            deltas[-1] = np.array([(yhat - rain) * w])

            for i in range(len(weights) - 2, -1, -1):
                a_i = activations[i + 1]
                relu_deriv = (a_i > 0).astype(float)   # ReLU derivative
                deltas[i] = (weights[i + 1].T @ deltas[i + 1]) * relu_deriv

            for i in range(len(weights)):
                grad_W[i] += np.outer(deltas[i], activations[i])
                if i == len(weights) - 1:
                    grad_b[i] += float(deltas[i].item())
                else:
                    grad_b[i] += deltas[i]

            if rain == y:
                successcount += 1
            else:
                failcount += 1

        # Apply averaged gradients once per batch
        batch_len = len(batch)
        for i in range(len(weights)):
            weights[i] -= n * grad_W[i] / batch_len
            biases[i]  -= n * grad_b[i] / batch_len

    success_history.append(successcount)
    fail_history.append(failcount)
    accuracy = successcount / (successcount + failcount)
    accuracy_history.append(accuracy)
    print(f"EPOCH {epoch} --- Accuracy: {accuracy * 100:.4f}%")

    if epoch >= min_epochs:
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_without_improvement = 0
            best_weights = [W.copy() for W in weights]
            best_biases = [b.copy() if isinstance(b, np.ndarray) else b for b in biases]
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} — no improvement for {patience} epochs")
                break

# --- Save weights ---
print("Saving final weights to file...")
weights = best_weights
biases = best_biases
print(f"Restoring best weights from accuracy: {best_accuracy * 100:.4f}%")
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
# plt.figure(figsize=(10, 5))
# plt.plot(success_history, label="Successes", color="green")
# plt.plot(fail_history, label="Failures", color="red")
# plt.xlabel("Epoch")
# plt.ylabel("Count per epoch")
# plt.title("MLP Training: Successes vs Failures per Epoch")
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.plot(accuracy_history, label="Accuracy", color="blue")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("MLP Training: Accuracy Over Time")
# plt.legend()
# plt.tight_layout()
# plt.show()