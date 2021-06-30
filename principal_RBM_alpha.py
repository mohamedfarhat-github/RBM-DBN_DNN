import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def to_num(alpha):
    if ord(alpha) >= 65:
        return ord(alpha)-55
    else :
        return ord(alpha)-48

def lire_alpha_digit(path, strSlct, nb_expl = 39):
    data = sio.loadmat(path)
    snip = None
    if isinstance(strSlct, str):
        for datum in strSlct:
            ind = to_num(datum)
            temp = np.concatenate(data['dat'][ind, 0:nb_expl], axis=1).flatten(order='F').reshape(320, nb_expl, order='F')
            if snip is not None:
                snip = np.hstack((snip, temp))
            else :
                snip = temp
    return snip.T

class RBM:
    
    def __init__(self, nb_v, nb_h):
        self.bias_a = np.zeros((1, nb_v))
        self.bias_b = np.zeros((1, nb_h))
        self.weights = 1e-1*np.random.randn(nb_v, nb_h)
        
    def sigmoid(self, data):
        return 1/(1+np.exp(-1*data))
    
    def softmax(self, data):
        return np.exp(data)/np.sum(np.exp(data), axis=1).reshape(-1, 1)

    def entree_sortie_RBM(self, v, act='sigmoid'):
        
        if act=='sigmoid':
            prob_h =  self.sigmoid(self.bias_b + v@self.weights)
        else:
            prob_h =  self.softmax(self.bias_b + v@self.weights)
        h = (np.random.uniform(0, 1, size=prob_h.shape) < prob_h) * 1
        
        return h, prob_h
    
    def sortie_entree_RBM(self, h, act='sigmoid'):
        
        if act=='sigmoid':
            prob_v =  self.sigmoid(self.bias_a + h@self.weights.T)
        else:
            prob_v =  self.softmax(self.bias_a + h@self.weights.T)
            
        v = (np.random.uniform(0, 1, size=prob_v.shape) < prob_v) * 1
        
        return v, prob_v
    
    def train_RBM(self, data, batch_size=50, n_epoch=10000, lr_rate=0.01, verbose=True):
        
        assert n_epoch > 0
        batch_size = batch_size if batch_size <= data.shape[1] else data.shape[1]
        
        for i in range(n_epoch):
            batch = data[np.random.choice(np.arange(data.shape[0]), batch_size), :]
            # Contrastive divergence (type I) 
            h_0, prob_h_0 = self.entree_sortie_RBM(batch)
            v_1, prob_v_1 = self.sortie_entree_RBM(h_0)
            h_1, prob_h_1 = self.entree_sortie_RBM(v_1)
            grad_a, grad_b = np.mean(batch - v_1, axis=0), np.mean(prob_h_0 - prob_h_1, axis=0)
            grad_weights = batch.T@prob_h_0 - v_1.T@prob_h_1
            self.bias_a += lr_rate*grad_a.reshape(1, -1)
            self.bias_b += lr_rate*grad_b.reshape(1, -1)
            self.weights += lr_rate/batch_size*grad_weights
            
            if verbose==True and i%1000==0:
                print("Iteration %d out of %d. Loss value is %.4f" %(i, n_epoch, 
                    np.sum((batch - v_1)**2)/batch_size))
                
        return self

    def generer_image_RBM(self, data_shape, nb_images=10, rows=20, cols=16, figsize=(8,6), max_iter=10000):
        
        data = np.random.rand(nb_images, data_shape)
        
        # Gibbs sampling
        for j in range(max_iter):
            data, _ = self.entree_sortie_RBM(data)
            data, _ = self.sortie_entree_RBM(data)
            
        im_cols = 3
        im_rows = nb_images // im_cols + nb_images % im_cols
        position = range(1,nb_images + 1)
        fig = plt.figure(1, figsize=figsize)
        
        for k in range(nb_images):
            ax = fig.add_subplot(im_rows, im_cols, position[k])
            ax.imshow(data[k, :].reshape(rows, cols, order='F'), cmap='gray')  
            
        return data
    
if __name__ == "__main__" :
   to_learn_chrs = "842CDF" # Characters that we want to learn
   data = lire_alpha_digit("./data/binaryalphadigs.mat", to_learn_chrs) # Getting the data
   alpha_rbm = RBM(data[0, :].shape[0], 250) # Constructing an RBM class instance
   alpha_rbm =  alpha_rbm.train_RBM(data, n_epoch=50000, verbose=True) # Training the RBM
   generated = alpha_rbm.generer_image_RBM(data[0, :].shape[0], 10, max_iter=30000)