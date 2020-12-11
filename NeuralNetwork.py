import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


class NeuralNetwork():
    '''
    size is for example [4,3,2] -> means we have
    4 neuron input
    3 neuron hidden layer
    2 neuron ouput
    '''

    def __init__(self, size, rand_seed=42):

        self.rand_seed = rand_seed
        np.random.seed(self.rand_seed)
        self.size = size
        ''' Inıtializing our all weights  '''
        weights = []
        for i in range(1, len(self.size)):
            weights.append(np.random.randn(self.size[i], self.size[i-1]))
        biases = []
        ''' All Bias values  are initilazing as vectors  '''
        for b in self.size[1:]
        biases.append(np.random.rand(b, 1))

    def forward(self, input):
        '''
        gets input x -> x*w -> v -> activation(v)
        '''
      a = input
      v_values = []
      y_values = [a]
      ''' Bütün inputları bütün w'lar ile çarparak
      y değerleri ve v değerleri
      delta hesabı için kaydediliyor'''
      for w, b in zip(self.weights, self.biases):
          v = np.dot(w, a) + b
          y = activation(v)
          v_values.append(v)
          y_values.append(y)
      return a, v_values, y_values

    def compute_deltas(self, v_values, y_true, y_pred):
      e = ( y_true - y_pred)
      delta_L =(e) * activation(v_values[-1], derivative=True)
      '''  gradyen_0 = e * sig'(v output)  '''

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

    def backward(self,deltas,v_values,y_values):


