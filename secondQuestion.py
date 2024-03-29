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
'''
yd_test = data_vectorizer(yd_test, 3)
yd_egitim = data_vectorizer(yd_egitim, 3) '''
x_test = data_vectorizer(x_test, 4)


allTestLosses = []
allTrainLosses = []

allTestAccuracies = []
allTrainAccuracies = []
for i in range(1):
    Network = NeuralNetwork.NeuralNetwork([4, 3, 3])
    epoch, loss, test_loss, test_accuracies, train_accuracies = Network.train(x_train=x_egitim, y_train=yd_egitim, x_test=x_test,
                                                                              y_test=yd_test, epochs=100, learning_rate=0.7, alfa=0.1, tqdm_=True, stop_error=0.0001)
    allTrainLosses.append(loss)
    allTestLosses.append(test_loss)
    allTestAccuracies.append(test_accuracies)
    allTrainAccuracies.append(train_accuracies)


# Gerekli Görsellemeler yapılarak sonuçlar yorumlanır
for i in range(len(allTrainLosses)):

    plt.plot(allTrainLosses[i])
    plt.title(' 3 kadar ara nöron için  Eğitim Karesel ortalama hata')
    plt.xlabel("Epoch sayısı")
    plt.ylabel("Toplam Karesel Ortalama Hata")
    plt.show()

    plt.plot(allTestAccuracies[i])
    plt.title(str(11-i) + '3 ara nöron için Test Datası Doğruluğu, ortalama = %' +
              str(np.mean(allTestAccuracies[i])))
    plt.xlabel("Data ")
    plt.ylabel("Test Datası İçin Doğruluk")
    plt.show()

    plt.plot(range(len(allTestLosses[i])), allTestLosses[i], "ro")
    plt.title('3 kadar ara nöron için Test Karesel ortalama hata')
    plt.xlabel("Data ")
    plt.ylabel("Toplam Karesel Ortalama Hata")
    plt.show()

    plt.plot(range(len(allTrainAccuracies[i])), allTrainAccuracies[i], "ro")
    plt.title(' 3 kadar ara nöron için Eğitim Datası Doğruluğu')
    plt.xlabel("Data ")
    plt.ylabel("Eğitim Datası için Doğruluk")
    plt.show()
