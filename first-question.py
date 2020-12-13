import NeuralNetwork
import numpy as np


""" Reading saved Data  """
X_train = np.load('x_egitim.npy')
y_desired = np.load('yd_egitim.npy')

X_test = np.load('x_test.npy')
y_test_desired = np.load('yd_test.npy')

Network = NeuralNetwork.NeuralNetwork([50, 3, 4])
History = Network.train(x_train=X_train, y_train=y_desired, x_test=X_test, y_test=y_test_desired, epochs=100, learning_rate=0.5, alfa=0.6, tqdm_=True)

print(History)