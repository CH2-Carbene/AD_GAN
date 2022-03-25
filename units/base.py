import numpy as np
import matplotlib as mpl
from units.globals import DEBUG
if not DEBUG: mpl.use("Agg")
import matplotlib.pyplot as plt
from functools import reduce

def fn_pipe(func_list):
    return lambda x: reduce(lambda inp,f:f(*inp),func_list,x)

def show(x):
    print(x,flush=True)

def show_process(pic_data,save_path=None):
    plt.figure(figsize=(15,10))
    x=range(0,len(pic_data)*200,200)
    # plt.yscale('log')
    plt.xlabel("Step")
    plt.plot(x,pic_data,label=["Tot loss","L2 loss","Gen loss","Disc loss"])
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path,dpi=100)
    if DEBUG:
        plt.show()
    plt.close()

def visualize(X,title="",save_path=None):
    """
    Visualize the image middle slices for each axis
    """
    if not isinstance(X,list):
        X=[X]
    # plt.title(title)
    plt.figure(figsize=(15,15))
    for i,x in enumerate(X):
        a,b,c = x.shape
        
        plt.subplot(len(X),3,3*i+1)
        plt.imshow(np.rot90(x[a//2, :, :]), cmap='gray')
        plt.axis('off')
        plt.subplot(len(X),3,3*i+2)
        plt.imshow(np.rot90(x[:, b//2, :]), cmap='gray')
        plt.axis('off')
        plt.subplot(len(X),3,3*i+3)
        plt.imshow(np.rot90(x[:, :, c//2]), cmap='gray')
        plt.axis('off')
    
    if save_path is not None:
        plt.savefig(save_path,dpi=100)
    if DEBUG:
        plt.show()
    plt.close()


def generate_images(model, test_input, tar,save_path=None,title=""):
    prediction = model(test_input, training=False)
    # plt.figure(figsize=(15, 15))
    # print(test_input[0].shape, tar[0].shape, prediction[0].shape)
    display_list = [test_input[0], tar[0], prediction[0][:,:,:,0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    # for i in range(3):
        # pass
        # plt.title(title[i])
    visualize(display_list,title=title,save_path=save_path)
        # Getting the pixel values in the [0, 1] range to plot.
    # plt.show()