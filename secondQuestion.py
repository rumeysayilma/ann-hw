#from sklearn.datasets import load_iris
import NeuralNetwork
import numpy as np
from matplotlib import pyplot as plt
import random

""" reading and manipulating data as we need it. """

f = open("Iris.csv", "r")
y_d = []
x = []

for row in f:
    rowsArray = row.split(',')
    rowsArray.pop(0)
    if(rowsArray[-1] == 'Iris-setosa\n'):

        y_d.append(np.array([1, 0, 0]).T)
        rowsArray.pop()
        x.append(np.array(rowsArray).astype(np.float).T)

    elif(rowsArray[-1] == 'Iris-versicolor\n'):

        y_d.append(np.array([0, 1, 0]).T)
        rowsArray.pop()
        x.append(np.array(rowsArray).astype(np.float).T)

    elif(rowsArray[-1] == 'Iris-virginica\n'):
        y_d.append(np.array([0, 0, 1]).T)
        rowsArray.pop()
        x.append(np.array(rowsArray).astype(np.float).T)


def datayi_karistir_test_ve_egitimi_ayir(x, y):
    c = list(zip(x, y))
    random.shuffle(c)
    (x_shuffled, yd_shuffled) = zip(*c)
    x_egitim = x_shuffled[0:120]
    x_test = x_shuffled[120:150]
    yd_egitim = yd_shuffled[0:120]
    yd_test = yd_shuffled[120:150]
    return x_egitim, x_test, yd_egitim, yd_test


x_egitim, x_test, yd_egitim, yd_test = datayi_karistir_test_ve_egitimi_ayir(
    x, y_d)

for i in range(len(x_egitim)):
    print(x_egitim[i])
    print('---------')
    print(yd_egitim[i])


def plot_history(history):
    n = history['epochs']
    plt.figure(figsize=(15, 5))
    n = 4000
    plt.plot(range(history['epochs'])[:n],
             history['train_loss'][:n], label='train_loss')

    plt.title('train  loss')
    plt.grid(1)
    plt.xlabel('epochs')
    plt.legend()

    plt.legend()


Network = NeuralNetwork.NeuralNetwork([4, 3, 3])
History = Network.train(x_train=x_egitim, y_train=yd_egitim, x_test=x_test,
                        y_test=yd_test, epochs=100, learning_rate=0.5, alfa=0.6, tqdm_=True)

print(History)

plot_history(History)
