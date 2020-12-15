from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

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

size = [50,3,4]

def cost_function(y_true, y_pred):
    n = y_pred.shape[1]
    cost = (1./(2*n)) * np.sum((y_true - y_pred) ** 2)
    return cost

#size boyutunda ağırlık kümesi oluşturulur.
#Bu küme her bir verinin eğitimi sonrasında güncellenerek yeni veri eğitime girer
weights = []
for i in range(1, len(size)):
    weights.append(np.random.rand(size[i], size[i-1]))

biases = []
for b in size[1:]:
    biases.append(np.random.rand(b, 1))

a = X_train
for i, (a, y_d) in enumerate(zip(a, y_desired)):
    #forward
    y_values = []
    v_values = []
    y_output_values = []
    y_values.append(a)
    #print(y_values)
    for z,(w,b) in enumerate(zip(weights, biases)):
        v = np.dot(w,y_values[z]) +b
        y = activation(v)
        v_values.append(v)
        #eğer son ağlıktaysak onu ayrı bir listede tutuyoruz.
        if z == (len(weights)-1):
            y_output_values.append(y)
            #y_values.append(y)
        else:
            y_values.append(y)
    #print(y_output_values, v_values, y_values)
    # Gradyenler bulunmaya başlandı

    #önce hata bulundu
    e = (np.reshape(y_d,(4,1)) - y_output_values[0])
    print('EEEEEEEE')
    print(e)

    #çıkış gradyanı bulundu ve tüm gradyanları içerecek kümeye eklendi
    gradients = [0] * (len(size)-1)
    print(',,,,,,,,,,,,,,,,,,,,,,,,,gradients')
    print(gradients)
    # grad1_0 = [2,3,4,5] , grad2_0 = [2,3,4,1] ,grad3_0.. grad4_0
    gradients[-1] =  e * activation(v_values[-1].T, derivative=True).T
    print(gradients[-1])
    print(',,,,,,,,,,,,,,,,,,,,,,,,')
    print(gradients)
    """ gradients[-1] = np.reshape(gradients[-1], (4,4)) """
    #gizli katmanı gradyanları bulundu ve gradyan kümesi tamamlandı
    for l in range(len(gradients) - 2, -1, -1):

        delta = np.dot(weights[l + 1].transpose(), gradients[l + 1]) * activation(v_values[l], derivative=True)
        gradients[l] = delta
    #print(gradients)

    #backpropagate
    #ağırlıkta oluşacak değişimlerin tutulduğu küme
    dW = []
    #biaste oluşacak değişikliklerin tutulduğu küme
    db = []
    gradients = [0] + gradients
    """ print(gradients) """
    for l in range(1, len(size)):
        print('****  '+str(l))
        print('y_values[l-1]')
        print(y_values[l-1])
        print(y_values[l-1].shape)
        print('------------------------')
        print(gradients[l])
        print(gradients[l].shape)

    #Bias ve ağırlıkta oluşacak değişimler bulunur
    for l in range(1, len(size)):

        """ 1 2 """
        dW_l = np.dot(gradients[l], y_values[l-1].transpose())
        db_l = gradients[l]
        dW.append(dW_l)
        db.append(np.expand_dims(db_l.mean(axis=1), 1))
        print(l)

        """
        for i in range(len(dW)):
        print(dW[i].shape)
        print('----------------------------------------------------------------')
        print(db[i].shape)
        """
    learning_rate=0.3
    alfa=0.4
    # weight update
    for i, (dw_each, db_each) in enumerate(zip(dW, db)):
        if i > 1:
            weights[i] = weights[i] - learning_rate * dw_each + alfa*(weights[i] - weights[i-1])
        else:
            weights[i] = weights[i] - learning_rate * dw_each

        biases[i] = biases[i] - learning_rate * db_each
    for i in range(len(weights)):
        print(' Weights ')
        print(weights[i])
        print('||||||||||||||||||||||||||||||||||||||||||||||||||||')
        print(weights[i].shape)

    for w, b in zip(weights, biases):
        z = np.dot(w, a) + b
        a = activation(z)
    predictions = (a > 0.5).astype(int)

    batch_y_train_pred = (predictions)
    train_losses = []
    train_loss = cost_function(y_d, batch_y_train_pred)
    train_losses.append(train_loss)
    print('train_losses')
    print(train_losses)
    """
    train_accuracies = []
    train_accuracy = accuracy_score(y_d.T, batch_y_train_pred.T)
    train_accuracies.append(train_accuracy)  """


