import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class NeuralNetwork(object):
    '''
    size is for example [4,3,2] -> means we have
    4 neuron input
    3 neuron hidden layer
    2 neuron ouput
    '''

    def __init__(self, size):

        self.size = size
        ''' Inıtializing our all weights  '''
        self.weights = []
        for i in range(1, len(self.size)):
            self.weights.append(np.random.rand(self.size[i], self.size[i-1]))
        self.biases = []
        ''' All Bias values  are initilazing as vectors  '''
        for b in self.size[1:]:
            self.biases.append(np.random.rand(b, 1))
        
    def __str__(self):
        return (str(self.weights[0]))
    def __repr__(self):
        return "<Test a:%s b:%s>" % (self.a, self.b)
    def forward(self, input):
        '''
         Bütün inputları bütün w'lar ile çarparak
        y değerleri ve v değerleri
        delta hesabı için kaydediliyor
        '''
        a = input
        v_values = []
        y_values=[]
        y_output_values =[]
        for i in range(len(a)):
            v_values.append([])
            y_values.append([])
            y_values[i].append(a[i]) 
            for z,(w, b) in enumerate(zip(self.weights, self.biases)):
                v = np.dot(w, y_values[i][z]) + b
                y = activation(v)
                v_values[i].append(v)
                if z == (len(self.weights)-1):
                    y_output_values.append(y)
                y_values[i].append(y)
        return(y_output_values, v_values, y_values)


    

    """ 
    inputs:
        v_values -> forward'dan hesapanan v değerleri.
        y_true -> datanın labellanmış yd değerleri.
        y_pred -> forwardda hesaplanan y değerleri.
    """
    def compute_deltas(self, v_values, y_desired, y_output_values):
        
        y_output_values = np.asarray(y_output_values).reshape(12,4)
        yd = y_desired.reshape(12,4)


        e = yd - y_output_values
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
            for l  in range(len(self.size)-1):
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
                delta = np.dot(self.weights[l + 1].transpose(), gradients[i][l + 1]) * activation(v_values[i][l], derivative=True)
                gradients[i][l].append(delta)
        return(gradients)
    def backpropagate(self, deltas, pre_activations, activations):
        """
        Applies back-propagation and computes the gradient of the loss
        w.r.t the weights and biases of the network

        Parameters:
        ---
        deltas: list of deltas computed by compute_deltas
        pre_activations: a list of pre-activations per layer
        activations: a list of activations per layer
        Returns:
        ---
        dW: list of gradients w.r.t. the weight matrices of the network
        db: list of gradients w.r.t. the biases (vectors) of the network
    
        """
        dW = []
        db = []
        deltas = [0] + deltas
        for l in range(1, len(self.size)):
            """ 1 2 """
            dW_l = np.dot(deltas[l], activations[l-1].transpose())
            db_l = deltas[l]
            dW.append(dW_l)
            db.append(np.expand_dims(db_l.mean(axis=1), 1))
        return dW, db    

def activation(z, derivative=False):

    if derivative:
        return activation(z) * (1 - activation(z))
    else:
        return 1 / (1 + np.exp(-z))

def cost_function(y_true, y_pred):
    """
    Computes the Mean Square Error between a ground truth vector and a prediction vector
    Parameters:
    ---
    y_true: ground-truth vector
    y_pred: prediction vector
    Returns:
    ---
    cost: a scalar value representing the loss
    """
    n = y_pred.shape[1]
    cost = (1./(2*n)) * np.sum((y_true - y_pred) ** 2)
    return cost

def cost_function_prime(y_true, y_pred):
    """
    Computes the derivative of the loss function w.r.t the activation of the output layer
    Parameters:
    ---
    y_true: ground-truth vector
    y_pred: prediction vector
    Returns:
    ---
    cost_prime: derivative of the loss w.r.t. the activation of the output
    shape: (n[L], batch_size)    
    """
    cost_prime = y_pred - y_true
    return cost_prime    