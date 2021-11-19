import numpy as np
import pandas as pd

model_weights = []

with open('weights_ResNet20_SHCUT.json', 'r') as f:
    for line in f:
        model_weights.append(np.array(line))
print((model_weights))

def weights_of_model(model):
    return [np.array(weights) for weights in model.get_weights()]


def get_random_weights(model):
    for w in weights_of_model(model):
        return [np.random.normal(loc=0, scale=1, size=w.shape)]