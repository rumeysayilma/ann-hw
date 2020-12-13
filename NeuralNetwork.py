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
        y_values.append(a)
    #print(y_values)  
        for z,(w,b) in enumerate(zip(self.weights, self.biases)):
            v = np.dot(w,y_values[z]) +b
            y = activation(v)
            v_values.append(v)
            #eğer son ağlıktaysak onu ayrı bir listede tutuyoruz.
            if z == (len(self.weights)-1):
                y_output_values.append(y)
                #y_values.append(y)
            else:
                y_values.append(y)
        return (y_output_values, v_values, y_values) 

    # Gradyenler bulunmaya başlandı
    def compute_gradients(self, v_values, y_d, y_output_values):

        #önce hata bulundu
        e = (np.reshape(y_d,(4,1)) - y_output_values[0])

        #çıkış gradyanı bulundu ve tüm gradyanları içerecek kümeye eklendi
        gradients = [0] * (len(self.size)-1)
        gradients[-1] =  e * activation(v_values[-1].T, derivative=True).T  

        #gizli katmanı gradyanları bulundu ve gradyan kümesi tamamlandı
        for l in range(len(gradients) - 2, -1, -1):          
            delta = np.dot(self.weights[l + 1].transpose(), gradients[l + 1]) * activation(v_values[l], derivative=True)
            gradients[l] = delta   

        """gradients bütün datalar için bütün gradientleri içerecek.yani her veri için giriş çıkış dahil 
        toplam katman sayısının 1 eksiği kadar gradyan burada tutulur.
        Bizim verimiz için 3 katman-1 --> 2 gradyan her veri için burada tutuluyor  """
        return(gradients)

    def backpropagate(self, gradients, v_values, y_values):

        dW = []
        db = []
        gradients = [0] + gradients  
        #Bias ve ağırlıkta oluşacak değişimler bulunur
        for l in range(1, len(self.size)):
            dW_l = np.dot(gradients[l], y_values[l-1].transpose())
            db_l = gradients[l]
            dW.append(dW_l)
            db.append(np.expand_dims(db_l.mean(axis=1), 1))
        return dW, db   

    def train(self, x_train, y_train, x_test, y_test, epochs, learning_rate, alfa, tqdm_=True):
        
        history_train_losses = []
        history_train_accuracies = []
        history_test_losses = []
        history_test_accuracies = []  

        if tqdm_:
            epoch_iterator = tqdm(range(epochs))
        else:
            epoch_iterator = range(epochs)

        for e in epoch_iterator:

            train_losses = []
            train_accuracies = []
            test_losses = []
            test_accuracies = []

            for i, (a, y_d) in enumerate(zip(x_train, y_train)): 

                y_output_values, v_values, y_values = self.forward(a)
                gradients = self.compute_gradients(v_values, y_d, y_output_values)
                dW, db = self.backpropagate(gradients, v_values, y_values)

                y_train_pred = y_output_values
                """ self.predict(a) """

                train_loss = cost_function(y_d, y_train_pred)
                train_losses.append(train_loss)
                #train_accuracy = accuracy_score(y_d.T, y_train_pred.T)
                #train_accuracies.append(train_accuracy)

                """                
                y_test_pred = self.predict(x_test)

                test_loss = cost_function(y_test, y_test_pred)
                test_losses.append(test_loss) 
                """
                """                 
                test_accuracy = accuracy_score(y_test.T, y_test_pred.T)
                test_accuracies.append(test_accuracy) """

                # weight update
                for i, (dw_each, db_each) in enumerate(zip(dW, db)):
                    if i > 1:
                        self.weights[i] = self.weights[i] - learning_rate * dw_each + alfa*(self.weights[i] - self.weights[i-1])
                    else:
                        self.weights[i] = self.weights[i] - learning_rate * dw_each 

                    self.biases[i] = self.biases[i] - learning_rate * db_each

                history_train_losses.append(np.mean(train_losses))
                #history_train_accuracies.append(np.mean(train_accuracies))           
                #history_test_losses.append(np.mean(test_losses))
                """ history_test_accuracies.append(np.mean(test_accuracies)) """

        history = {'epochs': epochs,
                   'train_loss': history_train_losses, 
                   }
        return history
    def predict(self, a):
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = activation(z)
        predictions = (a > 0.5).astype(int)
        return predictions


def activation(z, derivative=False):
    if derivative:
        return activation(z) * (1 - activation(z))
    else:
        return 1 / (1 + np.exp(-z))

def cost_function(y_true, y_pred):
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