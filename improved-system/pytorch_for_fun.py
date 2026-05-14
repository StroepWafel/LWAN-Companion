import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import copy
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ThreadPoolExecutor

# --- Config ---
DATAFILE       = "improved-system\\adelaide_weather_dataset.csv"
POPULATION     = 12
GENERATIONS    = 30
EPOCHS_PER_GEN = 20
SURVIVORS      = 4
MUTATION_RATE  = 0.3
BATCH_SIZE     = 256
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train(model: nn.Sequential, loader, lr: float, epochs: int) -> float:
    model.to(DEVICE)
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            criterion(model(xb), yb).backward()
            optimiser.step()
    model.eval()
    with torch.no_grad():
        X_dev = next(iter(loader))[0].to(DEVICE)  # just for shape — eval on full set below
    return 0.0  # placeholder, evaluated outside

def evaluate(model: nn.Sequential, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = (model(X.to(DEVICE)) >= 0.5).float()
        return (preds == y.to(DEVICE)).float().mean().item()

def random_genome() -> dict:
    return {
        "hidden_layers": [random.choice([4, 8, 12, 16, 24, 32]) for _ in range(random.randint(1, 4))],
        "lr":            random.choice([0.001, 0.005, 0.01, 0.02]),
        "dropout":       random.choice([0.0, 0.1, 0.2, 0.3]),
    }

def mutate(genome: dict) -> dict:
    g = copy.deepcopy(genome)
    if random.random() < MUTATION_RATE:
        g["lr"] = random.choice([0.001, 0.005, 0.01, 0.02])
    if random.random() < MUTATION_RATE:
        g["dropout"] = random.choice([0.0, 0.1, 0.2, 0.3])
    if random.random() < MUTATION_RATE:
        action = random.choice(["add", "remove", "resize"])
        if action == "add":
            g["hidden_layers"].append(random.choice([4, 8, 12, 16, 24, 32]))
        elif action == "remove" and len(g["hidden_layers"]) > 1:
            g["hidden_layers"].pop(random.randrange(len(g["hidden_layers"])))
        elif action == "resize":
            i = random.randrange(len(g["hidden_layers"]))
            g["hidden_layers"][i] = random.choice([4, 8, 12, 16, 24, 32])
    return g

def crossover(a: dict, b: dict) -> dict:
    child = copy.deepcopy(a)
    child["lr"]      = random.choice([a["lr"], b["lr"]])
    child["dropout"] = random.choice([a["dropout"], b["dropout"]])
    min_len = min(len(a["hidden_layers"]), len(b["hidden_layers"]))
    child["hidden_layers"] = [
        random.choice([a["hidden_layers"][i], b["hidden_layers"][i]])
        for i in range(min_len)
    ]
    return child

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # Load data
    df = pd.read_csv(DATAFILE)
    features = ["temperature", "humidity", "pressure", "wind_speed", "season_sin", "season_cos"]
    X = torch.tensor(df[features].values, dtype=torch.float32)
    y = torch.tensor(df["rain"].values, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X, y)

    # num_workers=0 required on Windows to avoid multiprocessing spawn issues
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    population = [random_genome() for _ in range(POPULATION)]
    best_accuracy_history = []
    best_genome = None
    best_accuracy = 0.0

    print(f"\nStarting evolution — {POPULATION} networks, {GENERATIONS} generations\n")

    for gen in range(GENERATIONS):
        print(f"Generation {gen + 1}/{GENERATIONS}")

        models = [build_network(g["hidden_layers"], g["dropout"]) for g in population]

        def train_one(args):
            idx, model, genome = args
            train(model, loader, genome["lr"], EPOCHS_PER_GEN)
            accuracy = evaluate(model, X, y)
            print(f"  [{idx+1:2d}] layers={genome['hidden_layers']} lr={genome['lr']} "
                  f"dropout={genome['dropout']} -> {accuracy*100:.2f}%")
            return accuracy, model, genome

        with ThreadPoolExecutor(max_workers=POPULATION) as ex:
            results = list(ex.map(train_one,
                           [(i, m, g) for i, (m, g) in enumerate(zip(models, population))]))

        results.sort(key=lambda x: x[0], reverse=True)
        gen_best_acc, gen_best_model, gen_best_genome = results[0]
        print(f"  Best: {gen_best_genome['hidden_layers']} -> {gen_best_acc*100:.2f}%\n")

        if gen_best_acc > best_accuracy:
            best_accuracy = gen_best_acc
            best_genome   = gen_best_genome
            torch.save(gen_best_model.state_dict(), "improved-system\\best_model.pt")
            with open("improved-system\\best_genome.json", "w") as f:
                json.dump(best_genome, f, indent=2)

        best_accuracy_history.append(gen_best_acc)

        survivors = [g for _, _, g in results[:SURVIVORS]]
        next_gen  = list(survivors)
        while len(next_gen) < POPULATION:
            a, b  = random.sample(survivors, 2)
            child = mutate(crossover(a, b))
            next_gen.append(child)
        population = next_gen

    print(f"\nEvolution complete.")
    if best_genome is not None:
        print(f"Best architecture: {best_genome['hidden_layers']}")
        print(f"Best lr: {best_genome['lr']}, dropout: {best_genome['dropout']}")
        print(f"Best accuracy: {best_accuracy*100:.2f}%")

    plt.figure(figsize=(10, 5))
    plt.plot([a * 100 for a in best_accuracy_history], color="blue",
             label="Best accuracy per generation")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy (%)")
    plt.title("Evolutionary MLP: Best Accuracy per Generation")
    plt.legend()
    plt.tight_layout()
    plt.show()