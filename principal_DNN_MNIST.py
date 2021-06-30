from principal_DBN_alpha import *
import tensorflow as tf
import numpy as np

def scale_and_read_img(data_set):
    snip = None
    data_std = (data_set - np.min(data_set))/(np.max(data_set) - np.min(data_set))
    for datum in data_std:
        temp = datum.flatten(order='F')
        if snip is not None:
            snip = np.vstack((snip, temp))
        else :
            snip = temp
    return snip

def transform_labels(labels):
    output = np.zeros((labels.shape[0], 10))
    for idx, label in enumerate(labels):
        output[idx, label] = 1
    return output

class MNIST_DNN(DNN):
    
    def __init__(self, n_layers):
        super().__init__(n_layers)
        
    def calcul_softmax(self, rbm_struct, data_in):
        _, h_p = rbm_struct.entree_sortie_RBM(data_in, act='softmax')
        return h_p
        
    def entree_sortie_reseau(self, data_in):
        outputs = [data_in]
        for rbm in self.RBM_list[:-1]:
            _ , data_in = rbm.entree_sortie_RBM(data_in)
            outputs.append(data_in)
        outputs.append(self.calcul_softmax(self.RBM_list[-1], data_in))
        return outputs
    
    def retropropagation(self, data, labels, batch_size=50, n_epoch=10000, lr_rate=0.01, verbose=True):
        assert n_epoch > 0
        batch_size = batch_size if batch_size <= data.shape[1] else data.shape[1]
        for i in range(n_epoch):
            batch_idx = np.random.choice(np.arange(data.shape[0]), batch_size)
            batch = data[batch_idx]
            batch_labels = labels[batch_idx]
            outputs = self.entree_sortie_reseau(batch)
            rbm = self.RBM_list[-1]
            c = (outputs[-1]-batch_labels)
            rbm.weights -= lr_rate/batch_size * outputs[-2].T @ c 
            rbm.bias_b -= lr_rate * np.mean(c, axis = 0) 
            for idx, rbm in reversed(list(enumerate(self.RBM_list[:-1]))):
                c = c@self.RBM_list[idx+1].weights.T*outputs[idx+1]*(1-outputs[idx+1])
                rbm.weights -= lr_rate/batch_size * outputs[idx].T @ c 
                rbm.bias_b -= lr_rate * np.mean(c, axis=0) 
            if verbose==True and i%1000==0:
                print("Iteration %d out of %d. CELoss value is %.4f" %(i, n_epoch,
                                                                -np.sum(batch_labels*np.log(outputs[-1]))/batch_size))
        return self
    
    def test_dnn(self, data, labels):
        for rbm in self.RBM_list[:-1]:
            _  ,  data = rbm.entree_sortie_RBM(data)
        preds = np.argmax(self.calcul_softmax(self.RBM_list[-1], data), axis=1)
        good_labels = 0
        for idx, pred in enumerate(preds):
            if pred==labels[idx]:
                good_labels+=1
        print("The percentage of false labeled data is ", 100*(labels.shape[0]-good_labels)/labels.shape[0])
        return 100*(labels.shape[0]-good_labels)/labels.shape[0]
        
        
if __name__ == "__main__" :
    
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    print('Transforming and standarising data...')
    data = scale_and_read_img(x_train) # This may take a while to execute
    test_data = scale_and_read_img(x_test)
    labels = transform_labels(y_train)
    
    layers = [data[0, :].shape[0], 700, 600, 500, 400, 300, 200, 100, 50, 10]
    test_dnn = MNIST_DNN(layers)
    print('begin unsupervised training')
    test_dnn = test_dnn.pretrain_DNN(data, batch_size=150, n_epoch=20000)
    print('Unsupervised training is done')
    
    print('begin supervised training')
    test_dnn = test_dnn.retropropagation(data, labels, lr_rate=0.1, batch_size=150, n_epoch=60000)
    print('Supervised training is done')
    test_dnn.test(test_data, y_test)
    
    test_dnn1 = MNIST_DNN(layers)
    print('begin unsupervised training')
    test_dnn1 = test_dnn1.retropropagation(data, labels, lr_rate=0.1, batch_size=150, n_epoch=60000)
    print('Unsupervised training is done')
    
    test_dnn1.test(test_data, y_test)







