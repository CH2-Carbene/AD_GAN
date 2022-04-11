import numpy as np
import matplotlib as mpl
import tensorflow as tf
from units.globals import DEBUG
if not DEBUG:
    mpl.use("Agg")
import matplotlib.pyplot as plt
from functools import reduce

def fn_pipe(func_list):
    return lambda x: reduce(lambda inp, f: f(inp), func_list, x)


def show(x):
    print(x, flush=True)


def show_process(pic_data, labels=["G_loss", "vox_loss", "gan_dis_loss", "D_loss"],save_path=None):
    plt.figure(figsize=(15, 10))
    x = range(0, len(pic_data)*200, 200)
    # plt.yscale('log')
    plt.xlabel("Step")
    plt.plot(x, pic_data, label=labels)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, dpi=100)
    if DEBUG:
        plt.show()
    plt.close()


def visualize(X, title="", save_path=None):
    """
    Visualize the image middle slices for each axis
    """
    if not isinstance(X, list):
        X = [X]
    # plt.title(title)
    plt.figure(figsize=(15, 5*len(X)))
    for i, x in enumerate(X):
        a, b, c = x.shape

        plt.subplot(len(X), 3, 3*i+1)
        plt.imshow(np.rot90(x[a//2, :, :]), cmap='gray')
        plt.axis('off')
        plt.subplot(len(X), 3, 3*i+2)
        plt.imshow(np.rot90(x[:, b//2, :]), cmap='gray')
        plt.axis('off')
        plt.subplot(len(X), 3, 3*i+3)
        plt.imshow(np.rot90(x[:, :, c//2]), cmap='gray')
        plt.axis('off')

    if save_path is not None:
        plt.savefig(save_path, dpi=100)
    if DEBUG:
        plt.show()
    plt.close()


def generate_images(model, inp, tar, save_path=None, title=""):
    fake = tf.reshape(model(inp[tf.newaxis,...,tf.newaxis], training=False), inp.shape)
    # plt.figure(figsize=(15, 15))
    # print(test_input[0].shape, tar[0].shape, prediction[0].shape)
    display_list = [inp, tar, fake]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    visualize(display_list, title=title, save_path=save_path)
    # Getting the pixel values in the [0, 1] range to plot.
    # plt.show()


def cyc_generate_images(G1, G2, imgA:tf.Tensor, imgB:tf.Tensor, save_path=None, title=""):
    fakeB = tf.reshape(G1(imgA[tf.newaxis,...,tf.newaxis], training=False), imgB.shape)
    cycA = tf.reshape(G2(fakeB[tf.newaxis,...,tf.newaxis], training=False), imgA.shape)
    fakeA = tf.reshape(G2(imgB[tf.newaxis,...,tf.newaxis], training=False), imgB.shape)
    cycB = tf.reshape(G1(fakeA[tf.newaxis,...,tf.newaxis], training=False), imgA.shape)

    # plt.figure(figsize=(15, 15))
    # print(test_input[0].shape, tar[0].shape, prediction[0].shape)
    display_list = [imgA,imgB,fakeA,fakeB,cycA,cycB]
    # title = ['Input Image', 'Ground Truth', 'Predicted Image']

    visualize(display_list, title=title, save_path=save_path)
