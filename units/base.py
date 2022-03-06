import numpy as np
import matplotlib.pyplot as plt

def visualize(X):
    """
    Visualize the image middle slices for each axis
    """
    a,b,c = X.shape
    
    plt.figure(figsize=(15,15))
    plt.subplot(131)
    plt.imshow(np.rot90(X[a//2, :, :]), cmap='gray')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(np.rot90(X[:, b//2, :]), cmap='gray')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(X[:, :, c//2], cmap='gray')
    plt.axis('off')