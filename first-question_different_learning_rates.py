import NeuralNetwork
import numpy as np
from matplotlib import pyplot as plt


""" Reading saved Data  """
X_train = np.load('x_egitim.npy')
y_desired = np.load('yd_egitim.npy')

X_test = np.load('x_test.npy')
y_test_desired = np.load('yd_test.npy')

allLosses = []

for i in range(10):
    Network = NeuralNetwork.NeuralNetwork([50, 12,  4])
    epoch, loss, test_loss, test_accuracies, train_accuracies = Network.train(x_train=X_train, y_train=y_desired, x_test=X_test,
                        y_test=y_test_desired, epochs=100, learning_rate=i*0.1, alfa=0.6, tqdm_=True)
    allLosses.append(loss)


for i in range(len(allLosses)):
    plt.plot(allLosses[i])
    plt.xlabel("Her veri için hata, Learning rate = "+ str(i*0.1))
    plt.ylabel("Toplam Karesel Ortalama Hata")
    plt.show()
