from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import NeuralNetwork

def activation(z, derivative=False):

    if derivative:
        return activation(z) * (1 - activation(z))
    else:
        return 1 / (1 + np.exp(-z))

""" Reading saved Data  """
X_train = np.load('x_egitim.npy')
y_desired = np.load('yd_egitim.npy')

X_test = np.load('x_test.npy')
y_test_desired = np.load('yd_test.npy')

network = NeuralNetwork.NeuralNetwork([50,4,4])


(y_output_values, v_values, y_values) =  network.forward(X_train)
""" print(y_output_values)
"""
gradients = network.compute_deltas(v_values,y_desired,y_output_values)
print(gradients)