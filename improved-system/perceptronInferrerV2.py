import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def train_neuron(x = np.array, w = np.array, b = 0):

    # Create output
    z = x.dot(w) + b
    y = 1 if (z >= 0) else 0

    # Update weights
    w = w + n * (rain - y) * x #type: ignore
    b = b + n * (rain - y) #type: ignore
    return(y, w, b)