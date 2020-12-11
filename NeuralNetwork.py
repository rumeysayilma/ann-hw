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
