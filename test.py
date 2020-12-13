from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

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

weights = []
for i in range(1, len(size)):
    weights.append(np.random.rand(size[i], size[i-1]))

biases = []
for b in size[1:]:
    biases.append(np.random.rand(b, 1))
a = X_train




"""
Forward

"""
v_values = []
y_values=[]
y_output_values = []
for i in range(len(a)):
    v_values.append([])
    y_values.append([])
    y_values[i].append(a[i]) 
    for z,(w, b) in enumerate(zip(weights, biases)):
        v = np.dot(w, y_values[i][z]) + b
        y = activation(v)
        v_values[i].append(v)
        if z == (len(weights)-1):
            y_output_values.append(y)
        y_values[i].append(y)
""" print(y_output_values, v_values, y_values)
"""

y_true = np.asarray(y_output_values).reshape(12,4)
yd = y_desired.reshape(12,4)
y_pred = y_values

e = yd - y_true
e = np.asarray(e)
""" v_values yukardan """

v_values = np.asarray(v_values)
gradyen_output = []

for i in range(len(v_values)):
    gradyen_output.append(e[i] * activation(v_values[i][-1].T, derivative=True))

"""gradients bütün datalar için bütün gradientleri içerecek.yani her veri için giriş çıkış dahil 
toplam katman sayısının 1 eksiği kadar gradyan burada tutulur.
Bizim verimiz için 3 katman-1 --> 2 gradyan her veri için burada tutuluyor  """

gradients =[]
for i in range(len(e)):
    gradients.append([])
    for l  in range(len(size)-1):
        gradients[i].append([])
      
""" gradyen outputs gradyenlere ekleniyor  """
for i  in range(len(e)):
    gradients[i][-1] = gradyen_output[i][0]
    
v_values =  np.asarray(v_values)
""" gradients  = [[[],[2,3,4,6]], ......] """
for i  in range(len(e)):
    """ 
    [[],[2,3,4,6]] 
    """
    print(gradients[i])
    for l in range(len(gradients[i]) - 2, -1, -1):
        delta = np.dot(weights[l + 1].transpose(), gradients[i][l + 1]) * activation(v_values[i][l], derivative=True)
        gradients[i][l].append(delta)
print(gradients[0])

dw = []
db = []




