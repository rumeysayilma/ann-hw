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
        weights = []
        for i in range(1, len(self.size)):
            weights.append(np.random.rand(self.size[i], self.size[i-1]))
        biases = []
        ''' All Bias values  are initilazing as vectors  '''
        for b in self.size[1:]:
            biases.append(np.random.rand(b, 1))

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
            for z,(w, b) in enumerate(zip(weights, biases)):
                v = np.dot(w, y_values[i][z]) + b
                y = activation(v)
                v_values[i].append(v)
                if z == (len(weights)-1):
                    y_output_values.append(y)
                y_values[i].append(y)
        return(y_output_values, v_values, y_values)


    

 
    def compute_deltas(self, v_values, y_true, y_pred):

        e = (y_true - y_pred)
        delta_L = (e) * activation(v_values[-1], derivative=True)
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

    def backward(self, deltas, v_values, y_values):
        """
        Geri dönüş yaparken Her ağırlığın ve bias'in ne kadar güncelleneceğini
        hesaplayıp bunları dW ve db listelerine ekliyoruz.
        """
        dW = []
        db = []
        deltas = [0] + deltas
        """ Size'ın bir eksiğine kadar gidiyorlar. """
        for l in range(1, len(self.size)):
            """ Delta ile  bir önceki katmandaki çıkışı -> y değerlerini çarpıyoruz. """
            dW_l = np.dot(deltas[l], y_values[l-1].transpose())
            db_l = deltas[l]
            dW.append(dW_l)
            db.append(np.expand_dims(db_l.mean(axis=1), 1))
        return dW, db

    def plot_decision_regions(self, X, y, iteration, train_loss, val_loss, train_acc, val_acc, res=0.01):
        """
        Plots the decision boundary at each iteration (i.e. epoch) in order to inspect the performance
        of the model

        Parameters:
        ---
        X: the input data
        y: the labels
        iteration: the epoch number
        train_loss: value of the training loss
        val_loss: value of the validation loss
        train_acc: value of the training accuracy
        val_acc: value of the validation accuracy
        res: resolution of the plot
        Returns:
        ---
        None: this function plots the decision boundary
        """
        X, y = X.T, y.T
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                             np.arange(y_min, y_max, res))

        Z = self.predict(np.c_[xx.ravel(), yy.ravel()].T)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.5)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1),  alpha=0.2)
        message = 'iteration: {} | train loss: {} | val loss: {} | train acc: {} | val acc: {}'.format(iteration,
                                                                                                       train_loss,
                                                                                                       val_loss,
                                                                                                       train_acc,
                                                                                                       val_acc)
        plt.title(message)

    def train(self, X, y, batch_size=1, epochs, learning_rate, validation_split=0.2, print_every=1, tqdm_=True,  plot_every=None):
        """
        Parameters:
        ---
        X: input data
        y: input labels
        batch_size: number of data points to process in each batch
        epochs: number of epochs for the training
        learning_rate: value of the learning rate
        validation_split: percentage of the data for validation
        print_every: the number of epochs by which the network logs the loss and accuracy metrics for train and validations splits
        plot_every: the number of epochs by which the network plots the decision boundary
        Returns:
        ---
        history: dictionary of train and validation metrics per epoch
            train_acc: train accuracy
            test_acc: validation accuracy
            train_loss: train loss
            test_loss: validation loss
        This history is used to plot the performance of the model
        """
        history_train_losses = []
        history_train_accuracies = []
        history_test_losses = []
        history_test_accuracies = []

        x_train, x_test, y_train, y_test = train_test_split(
            X.T, y.T, test_size=validation_split, )
        x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T

        """ Starting  """
        for e in range(epoch_iterator):
            if x_train.shape[1] % batch_size == 0:
                n_batches = int(x_train.shape[1] / batch_size)
            else:
                n_batches = int(x_train.shape[1] / batch_size) - 1

            x_train, y_train = shuffle(x_train.T, y_train.T)
            x_train, y_train = x_train.T, y_train.T

            batches_x = [x_train[:, batch_size*i:batch_size *
                                 (i+1)] for i in range(0, n_batches)]
            batches_y = [y_train[:, batch_size*i:batch_size *
                                 (i+1)] for i in range(0, n_batches)]

            train_losses = []
            train_accuracies = []

            test_losses = []
            test_accuracies = []
            """
            Her  epochtaki w değişimini eklemek için aynı boyutta 0lardan oluşan
            bir liste oluşturuyorlar.
            """
            dw_per_epoch = [np.zeros(w.shape) for w in self.weights]
            db_per_epoch = [np.zeros(b.shape) for b in self.biases]

            for batch_x, batch_y in zip(batches_x, batches_y):
                """ forward fonksyionu giriş alıp ağırlıklarla çarpıp v ve y çıkartıyor"""
                batch_y_pred, pre_activations, activations = self.forward(
                    batch_x)
                """ v leri y leri,asıl ydleri alıp gradyenleri alıyoruz."""
                deltas = self.compute_deltas(
                    pre_activations, batch_y, batch_y_pred)
                """
                       bulduğumz deltalarla  geri dönüp
                       deltas,v_values,y_values alarak dw db yani ağırlıkların ne kadar
                       değişeceğini hesaplıyor.
                       """
                dW, db = self.backward(
                    deltas, pre_activations, activations)
                for i, (dw_i, db_i) in enumerate(zip(dW, db)):
                    """ her epcohtaki değişen ağırlık miktarını ekliyoruz.  """
                    dw_per_epoch[i] += dw_i / batch_size
                    db_per_epoch[i] += db_i / batch_size

                batch_y_train_pred = self.predict(batch_x)

                train_loss = cost_function(batch_y, batch_y_train_pred)
                train_losses.append(train_loss)
                train_accuracy = accuracy_score(
                    batch_y.T, batch_y_train_pred.T)
                train_accuracies.append(train_accuracy)

                batch_y_test_pred = self.predict(x_test)

                test_loss = cost_function(y_test, batch_y_test_pred)
                test_losses.append(test_loss)
                test_accuracy = accuracy_score(
                    y_test.T, batch_y_test_pred.T)
                test_accuracies.append(test_accuracy)

            # weights are updating for each batch for each weights ( weihgts are storied for each weight array )
            for i, (dw_epoch, db_epoch) in enumerate(zip(dw_per_epoch, db_per_epoch)):
                self.weights[i] = self.weights[i] - learning_rate * dw_epoch
                self.biases[i] = self.biases[i] - learning_rate * db_epoch

            history_train_losses.append(np.mean(train_losses))
            history_train_accuracies.append(np.mean(train_accuracies))

            history_test_losses.append(np.mean(test_losses))
            history_test_accuracies.append(np.mean(test_accuracies))

            if not plot_every:
                if e % print_every == 0:
                    print('Epoch {} / {} | train loss: {} | train accuracy: {} | val loss : {} | val accuracy : {} '.format(
                        e, epochs, np.round(np.mean(train_losses), 3), np.round(
                            np.mean(train_accuracies), 3),
                        np.round(np.mean(test_losses), 3),  np.round(np.mean(test_accuracies), 3)))
            else:
                if e % plot_every == 0:
                    self.plot_decision_regions(x_train, y_train, e,
                                               np.round(
                                                   np.mean(train_losses), 4),
                                               np.round(
                                                   np.mean(test_losses), 4),
                                               np.round(
                                                   np.mean(train_accuracies), 4),
                                               np.round(
                                                   np.mean(test_accuracies), 4),
                                               )
                    plt.show()
                    display.display(plt.gcf())
                    display.clear_output(wait=True)

        self.plot_decision_regions(X, y, e,
                                   np.round(np.mean(train_losses), 4),
                                   np.round(np.mean(test_losses), 4),
                                   np.round(np.mean(train_accuracies), 4),
                                   np.round(np.mean(test_accuracies), 4),
                                   )

        history = {'epochs': epochs,
                   'train_loss': history_train_losses,
                   'train_acc': history_train_accuracies,
                   'test_loss': history_test_losses,
                   'test_acc': history_test_accuracies
                   }
        return history

    def plot_graph():
        print('We are gonna plot this')
