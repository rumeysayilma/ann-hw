import NeuralNetwork
import numpy as np
from matplotlib import pyplot as plt


""" Reading saved Data  """
X_train = np.load('x_egitim.npy')
y_desired = np.load('yd_egitim.npy')

X_test = np.load('x_test.npy')
y_test_desired = np.load('yd_test.npy')

allTestLosses = []
allTrainLosses = []

for i in range(10):
    Network = NeuralNetwork.NeuralNetwork([50, 12-i,  4])
    epoch,loss,test_loss = Network.train(x_train=X_train, y_train=y_desired, x_test=X_test,
                            y_test=y_test_desired, epochs=100, learning_rate=0.6, alfa=0.6, tqdm_=True)
    allTestLosses.append(test_loss)
    allTrainLosses.append(loss)

for i in range(len(allTrainLosses)):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Sol Eğitim - Sağ Test ,Karesel ortalama hata')
    ax1.plot(allTrainLosses[i])
    ax2.axis([0,8, 0, 1])
    ax2.plot(allTestLosses[i])
    plt.show()