from principal_DNN_MNIST import *
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# Reloading modules just in case

if __name__ == "__main__" :
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    data_mnist = scale_and_read_img(x_train)
    test_data = scale_and_read_img(x_test) # scaling test data
    labels = transform_labels(y_train) #transforming labels
    
    layers = [data_mnist[0, :].shape[0], 700, 600, 500, 400, 300, 200, 100, 50, 10]
    
    test_dnn = MNIST_DNN(layers)
    print('begin unsupervised training')
    test_dnn = test_dnn.pretrain_DNN(data_mnist, batch_size=150, n_epoch=15000)
    print('Unsupervised training is done')
    
    print('begin supervised training')
    test_dnn = test_dnn.retropropagation(data_mnist, labels, lr_rate=0.1, batch_size=150, n_epoch=30000)
    print('Supervised training is done')
    
    test_dnn.test_dnn(test_data, y_test)
    
    generated = test_dnn.generer_image_DBN(data_mnist[0, :].shape[0], 10, rows=28, cols=28, max_iter=15000)
    
    ### L'effet du nombre de couches sur la classification
        
    hidden_mult = 6
    layers_list = []
    layers=[200]
    for i in range(2,hidden_mult+1):
        layers_list.append(i*layers)
    for layers in layers_list:
        layers.insert(0, data_mnist[0, :].shape[0])
        layers.append(10)
    
    test_pretrain, test_retro = [], []
    for layer in layers_list:
        dnn_pretrain = MNIST_DNN(layer)
        dnn_retro = MNIST_DNN(layer)
        
        print('Training pretrained version')
        print('begin unsupervised training')
        dnn_pretrain = dnn_pretrain.pretrain_DNN(data_mnist, batch_size=150, n_epoch=15000)
        print('Unsupervised training is done')
        
        print('Begin supervised training')
        dnn_pretrain = dnn_pretrain.retropropagation(data_mnist, labels, lr_rate=0.1, batch_size=150, n_epoch=40000)
        print('Supervised training is done')
        
        print('Training non pretrained version')
        print('Begin supervised training')
        dnn_retro = dnn_retro.retropropagation(data_mnist, labels, lr_rate=0.1, batch_size=150, n_epoch=40000)
        print('Supervised training is done')
        
        test_pretrain.append(dnn_pretrain.test_dnn(test_data, y_test))
        test_retro.append(dnn_retro.test_dnn(test_data, y_test))
    
    plt.figure(figsize=(15,6))
    plt.plot([i for i in range(2,hidden_mult+1)], test_pretrain, '--g', label='Modèle pré-entraîné') # dashed cyan
    plt.plot([i for i in range(2,hidden_mult+1)], test_retro, '-.k', label='Modèle non pré-entraîné') # dashdot black
    plt.grid()
    plt.legend(loc="upper right")
    plt.title('Pourcentage d\'erreurs de classification par rapport au numéro de couche cachée');
    plt.xlabel('Numéro de couche cachée')
    plt.ylabel('Pourcentage d\'erreurs de classification');
    
    plt.figure(figsize=(15,6))
    plt.plot([i for i in range(2,hidden_mult+1)], test_pretrain, '--g', label='Modèle pré-entraîné') # dashed cyan
    plt.plot([i for i in range(2,hidden_mult+1)], test_retro, '-.k', label='Modèle non pré-entraîné') # dashdot black
    plt.ylim(0,4)
    plt.grid()
    plt.legend(loc="upper right")
    plt.title('Pourcentage d\'erreurs de classification par rapport au numéro de couche cachée');
    plt.xlabel('Numéro de couche cachée')
    plt.ylabel('Pourcentage d\'erreurs de classification');
    
    ### L'effet du nombre de neurones de la couche cachée sur la classification
        
    hidden_mult = 7
    layers_list = []
    for i in range(2,hidden_mult+1):
        layers=[100, 100]
        for j in range(len(layers)):
            layers[j] *= i
        layers_list.append(layers)
    for layers in layers_list:
        layers.insert(0, data_mnist[0, :].shape[0])
        layers.append(10)
    
    test_pretrain_size, test_retro_size = [], []
    for layer in layers_list:
        dnn_pretrain = MNIST_DNN(layer)
        dnn_retro = MNIST_DNN(layer)
        
        print('Training pretrained version')
        print('begin unsupervised training')
        dnn_pretrain = dnn_pretrain.pretrain_DNN(data_mnist, batch_size=150, n_epoch=15000)
        print('Unsupervised training is done')
        
        print('Begin supervised training')
        dnn_pretrain = dnn_pretrain.retropropagation(data_mnist, labels, lr_rate=0.1, batch_size=150, n_epoch=30000)
        print('Supervised training is done')
        
        print('Training non pretrained version')
        print('Begin supervised training')
        dnn_retro = dnn_retro.retropropagation(data_mnist, labels, lr_rate=0.1, batch_size=150, n_epoch=30000)
        print('Supervised training is done')
        
        test_pretrain_size.append(dnn_pretrain.test_dnn(test_data, y_test))
        test_retro_size.append(dnn_retro.test_dnn(test_data, y_test))
    
    plt.figure(figsize=(15,6))
    plt.plot([100*i for i in range(2,hidden_mult+1)], test_pretrain_size, '--g', label='Modèle pré-entraîné') # dashed cyan
    plt.plot([100*i for i in range(2,hidden_mult+1)], test_retro_size, '-.k', label='Modèle non pré-entraîné') # dashdot black
    plt.ylim(0,4)
    plt.grid()
    plt.legend(loc="upper right")
    plt.title('Pourcentage d\'erreurs de classification par rapport au nombre de neurones dans les couches cachées');
    plt.xlabel('Nombre de neurones dans les couches cachées')
    plt.ylabel('Pourcentage d\'erreurs de classification');
    
    ### L'effet du L'effet du nombre d'échantillons d'apprentissage sur l'erreur de classification
        
    sizes = [1000, 3000, 7000, 10000, 30000]
    data_list = []
    labels_list = []
    layers = [data_mnist[0, :].shape[0], 200, 200, 10]
    for size in sizes:
        idx = np.random.choice(np.arange(data_mnist.shape[0]), size)
        data_sample = data[idx]
        labels_sample = labels[idx]
        data_list.append(data_sample)
        labels_list.append(labels_sample)
    labels_list.append(labels)
    data_list.append(data_mnist)
    
    test_pretrain_data, test_retro_data = [], []
    for i in range(len(sizes)):
        dnn_pretrain = MNIST_DNN(layers)
        dnn_retro = MNIST_DNN(layers)
        
        print('Training pretrained version')
        print('begin unsupervised training')
        dnn_pretrain = dnn_pretrain.pretrain_DNN(data_list[i], batch_size=150, n_epoch=15000)
        print('Unsupervised training is done')
        
        print('Begin supervised training')
        dnn_pretrain = dnn_pretrain.retropropagation(data_list[i], labels_list[i], lr_rate=0.1, batch_size=150, n_epoch=30000)
        print('Supervised training is done')
        
        print('Training non pretrained version')
        print('Begin supervised training')
        dnn_retro = dnn_retro.retropropagation(data_list[i], labels_list[i], lr_rate=0.1, batch_size=150, n_epoch=30000)
        print('Supervised training is done')
        
        test_pretrain_data.append(dnn_pretrain.test_dnn(test_data, y_test))
        test_retro_data.append(dnn_retro.test_dnn(test_data, y_test))
    
    plt.figure(figsize=(15,6))
    plt.plot(sizes, test_pretrain_data, '--g', label='Modèle pré-entraîné') # dashed cyan
    plt.plot(sizes, test_retro_data, '-.k', label='Modèle non pré-entraîné') # dashdot black
    plt.ylim(0,4)
    plt.grid()
    plt.legend(loc="upper right")
    plt.title('Pourcentage d\'erreurs de classification par rapport à la taille de l\'ensemble de données d\'entraînement.');
    plt.xlabel('Taille de l\'ensemble de données d\'entraînement')
    plt.ylabel('Pourcentage d\'erreurs de classification');