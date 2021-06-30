from principal_RBM_alpha import RBM, lire_alpha_digit
import numpy as np
import matplotlib.pyplot as plt

class DNN :
    
    def __init__(self, n_layers):
        self.RBM_list = [RBM(lyr_prd, lyr_nxt) for lyr_prd, lyr_nxt in zip(n_layers, n_layers[1:])]
        
    def pretrain_DNN(self, data, batch_size=50, n_epoch=10000, lr_rate=0.01, verbose=True):
        for ind, rbm in enumerate(self.RBM_list):
            print('Training layer ', ind)
            rbm = rbm.train_RBM(data, batch_size=batch_size, n_epoch=n_epoch, lr_rate=lr_rate, verbose=verbose)
            _, data  = rbm.entree_sortie_RBM(data)
        return self
    
    def generer_image_DBN(self, data_shape, nb_images=10, rows=20, cols=16, figsize=(8,6), max_iter=10000):
        
        data = np.random.binomial(1, 0.5, size=(nb_images, data_shape))
        
        for itr in range(max_iter):
            for rbm in self.RBM_list[:-1]:
                data, _ = rbm.entree_sortie_RBM(data)
                
            data, _ = self.RBM_list[-1].entree_sortie_RBM(data)
            
            for rbm in reversed(self.RBM_list[1:]):
                _, data= rbm.sortie_entree_RBM(data)
                
            data, _ = self.RBM_list[0].sortie_entree_RBM(data)
                
        im_cols = 3
        im_rows = nb_images // im_cols + nb_images % im_cols
        position = range(1,nb_images + 1)
        fig = plt.figure(1, figsize=figsize)
        for k in range(nb_images):
            ax = fig.add_subplot(im_rows, im_cols, position[k])
            ax.imshow(data[k, :].reshape(rows, cols, order='F'), cmap='gray')      
        return data
    
if __name__ == "__main__" :
    to_learn_chrs = "524ABN"
    data = lire_alpha_digit("./data/binaryalphadigs.mat", to_learn_chrs)
    layers = [data[0, :].shape[0], 300, 250, 200, 150, 100]
    alpha_dbn = DNN(layers)
    alpha_dbn =  alpha_dbn.pretrain_DNN(data, n_epoch=20000, verbose=True)
    generated = alpha_dbn.generer_image_DBN(data[0, :].shape[0], 10, max_iter=15000)