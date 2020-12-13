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
""" 
def compute_deltas(self, v_values, y_true, y_pred):
"""
"""  """
e = yd- y_true
e = np.asarray(e)
""" v_values yukardan """
v_values = np.asarray(v_values)
gradyen_output = []

for i in range(len(v_values)):
    gradyen_output.append(e[i] * activation(v_values[i][-1].T, derivative=True))
    print(e[i])
    print('****')
    print(v_values[i][-1])

"""gradients bütün datalar için bütün gradientleri içerecek.  """
gradients =[]
for i in range(len(e)):
    gradients.append([])



gradients[-1] = gradyen_output
print(gradients)

print('--------------------------------')
print(gradyen_output)
for i  in range(len(e)):
    gradients.append([])
    for l in range(len(gradients) - 2, -1, -1):
        delta = np.dot(weights[i][l + 1].transpose(), gradients[i][l + 1]) * activation(v_values[l], derivative=True)
        gradients[i].append(delta)
print(gradients)

""" 
    def compute_deltas(self, v_values, y_true, y_pred):


        ''' katman sayısının bir eksiği kadar delta olacak'''
        deltas = [0] * (len(self.size) - 1)
        '''gradyen_0 dan başa giderek gradyenler bulunacak
      sonuncusu grandyen_0 '''
        deltas[-1] = delta_L
        ''' zaten son katmandaki deltayı bulduk
      katman sayısı-1 kadar delta olacaktı.
      yani deltas-2 kadar delta kaldı bulacağımız.
      yani delta -2'den 0'a kadar (0 dahil )
      listenin sonundan başa giderek loop yapıyoruz.
      ve bir önceki katmandaki deltayı bulup listeye ekliyoruz.
         '''

        for l in range(len(deltas) - 2, -1, -1):
            delta = np.dot(self.weights[l + 1].transpose(), deltas[l + 1]
                           ) * activation(v_values[l], derivative=True)
            deltas[l] = delta
        return deltas

""" 
"""
print(v_values[1].shape)
'''  gradyen_0 = e * sig'(v output)  '''


print(e) """

