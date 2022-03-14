import numpy as np
import matplotlib.pyplot as plt
from units.globals import DEBUG

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
    prediction = model(test_input, training=True)
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