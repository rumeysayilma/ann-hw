#from sklearn.datasets import load_iris
import NeuralNetwork
import numpy as np
from matplotlib import pyplot as plt
import random
import math
import random

""" Generating Billing Data. """
y_d = [0, 0]
x = []

for i in range(2, 100):
    a = []
    a.append(0.8 - 0.5 * math.exp(-((y_d[i-1])**2)))
    a.append((0.3 + 0.9 * math.exp(-(y_d[i-1]**2)))*(y_d[i-2]))
    a.append(0.1 * math.pi * math.sin(math.pi*(y_d[i-1])))
    a = np.array(a)
    x.append(a)
    y_dvalue = []
    y_d.append((a[0] + a[1]+a[2] + random.random()))


def data_vectorizer(data_to_vector, n):
    final = [a.reshape((n, 1)) for a in data_to_vector]
    return final


def datayi_karistir_test_ve_egitimi_ayir(x, y):
    c = list(zip(x, y))
    random.shuffle(c)
    (x_shuffled, yd_shuffled) = zip(*c)
    x_egitim = x_shuffled[0:80]
    x_test = x_shuffled[80:100]
    yd_egitim = yd_shuffled[0:80]
    yd_test = yd_shuffled[80:100]
    return x_egitim, x_test, yd_egitim, yd_test


x_egitim, x_test, yd_egitim, yd_test = datayi_karistir_test_ve_egitimi_ayir(
    x, y_d)

x_egitim = data_vectorizer(x_egitim, 3)
'''
yd_test = data_vectorizer(yd_test, 1)
yd_egitim = data_vectorizer(yd_egitim, 1)
'''
x_test = data_vectorizer(x_test, 3)
print(y_d)

allTestLosses = []
allTrainLosses = []

allTestAccuracies = []
allTrainAccuracies = []
for i in range(10):
    Network = NeuralNetwork.NeuralNetwork([3, 2,  1])
    epoch, loss, test_loss, test_accuracies, train_accuracies = Network.train(x_train=x_egitim, y_train=yd_egitim, x_test=x_test,
                                                                              y_test=yd_test, epochs=50, learning_rate=0.7, alfa=0.6, tqdm_=True, stop_error=0.0001)
    allTrainLosses.append(test_loss)
    allTestLosses.append(loss)
    allTestAccuracies.append(test_accuracies)
    allTrainAccuracies.append(train_accuracies)
print(allTrainLosses[0])
print(len(allTrainLosses[0]))

# Gerekli Görsellemeler yapılarak sonuçlar yorumlanır
for i in range(len(allTrainLosses)):

    plt.plot(allTestAccuracies[i])
    plt.title('Test Datası Doğruluğu, ortalama = %' +
              str(np.mean(allTestAccuracies[i])))
    plt.xlabel("Data ")
    plt.ylabel("Test Datası İçin Doğruluk")
    plt.show()

    plt.plot(range(len(allTrainLosses[i])), allTrainLosses[i], "ro")
    plt.title('Test Karesel ortalama hata')
    plt.xlabel("Data ")
    plt.ylabel("Toplam Karesel Ortalama Hata")
    plt.show()

    plt.plot(range(len(allTrainAccuracies[i])), allTrainAccuracies[i], "ro")
    plt.title( ' Eğitim Datası Doğruluğu , ortalama= %' +
              str(np.mean(allTrainAccuracies[i])))
    plt.xlabel("Data ")
    plt.ylabel("Eğitim Datası için Doğruluk")
    plt.show()
