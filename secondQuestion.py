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


def data_vectorizer(data_to_vector, n):
    final = [a.reshape((n, 1)) for a in data_to_vector]
    return final


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
x_egitim = data_vectorizer(x_egitim, 4)
yd_test = data_vectorizer(yd_test, 3)
yd_egitim = data_vectorizer(yd_egitim, 3)
x_test = data_vectorizer(x_test, 4)


allTestLosses = []
allTrainLosses = []
allTestAccuracies = []
for i in range(10):
    Network = NeuralNetwork.NeuralNetwork([4, 11-i, 3])
    epoch, loss, test_loss, test_accuracies = Network.train(x_train=x_egitim, y_train=yd_egitim, x_test=x_test,
                                                            y_test=yd_test, epochs=10, learning_rate=0.7, alfa=0.6, tqdm_=True)
    allTrainLosses.append(test_loss)
    allTestLosses.append(loss)
    allTestAccuracies.append(test_accuracies)
print(allTrainLosses[0])
print(len(allTrainLosses[0]))


for i in range(len([1, 2, 34])):
    plt.plot(allTrainLosses[i])
    plt.title(str(11-i) + ' kadar ara nöron için  Eğitim Karesel ortalama hata')
    plt.xlabel("Epoch sayısı")
    plt.ylabel("Toplam Karesel Ortalama Hata")
    plt.show()

    plt.plot(allTestAccuracies[i])
    plt.title(str(11-i) + ' Test Accuracy for each data')
    plt.xlabel("Data ")
    plt.ylabel("Toplam Karesel Ortalama Hata")
    plt.show()

    plt.scatter(range(len(allTrainLosses)), allTrainLosses[i])
    plt.title(str(11-i) + ' kadar ara nöron için Test Karesel ortalama hata')
    plt.xlabel("Data ")
    plt.ylabel("Toplam Karesel Ortalama Hata")
    plt.show()
