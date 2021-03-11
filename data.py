import tensorflow.keras.datasets as datasets
import tensorflow.keras.utils as utils
import matplotlib.pyplot as plt
import numpy as np


def get_mnist_data(to_cat=utils.to_categorical):
    (X_train, Y_train), (X_test, Y_test) = datasets.mnist.load_data()
    Y_train, Y_test = [to_cat(x) for x in [Y_train, Y_test]]
    return [X_train, Y_train], [X_test, Y_test]


flatten_spatial_imgs = lambda X: np.reshape(X, (X.shape[0], np.prod(X.shape[1:])))
simple_Normalize = lambda X: (X - X.min()) / (X.max() - X.min())

def plt_imgs(imgs, y_act, y_pred=None, fig_dim=(3,3), intr =np.argmax,sup="visialized Trained Waight", cmap='seismic' ) :
    fig, axis = plt.subplots( fig_dim[0], fig_dim[1])
    fig.suptitle(sup)
    for i, ax in enumerate( axis.flat):
        img = np.reshape(imgs[i],(28, 28))
        ax.imshow(img, cmap = cmap )
        if y_pred is None :
            xlabel = 'label %d'%(intr(y_act[i]))
        else:
            xlabel = 'label %d, pred %d'%(intr(y_act[i]), intr(y_pred[i]))
        ax.set_xlabel(xlabel)
        ax.set_xticks([]); ax.set_yticks([])

def get_prepro_imgs(data_gen = get_mnist_data,
                    fltter = flatten_spatial_imgs,
                    Normalizer = simple_Normalize):
    train, test = data_gen()
    train[0], test[0] = [fltter(x) for x in [train[0], test[0]]]
    train[0], test[0] = [Normalizer(x) for x in [train[0], test[0]]]
    return train, test



if __name__ == '__main__':
    train, test = get_mnist_data()
    plt_imgs(train[0], train[1], test[1])
