### image argument & preprocession functions
from scipy.ndimage.interpolation import affine_transform
import elasticdeform,scipy
import numpy as np

def Patch_extration(psize=(128,128,128)):
    def patch_extraction(x,y):
        """
        3D patch extraction
        """
        # xlen,ylen,zlen=x.shape
        st_range=np.array(x.shape)-np.array(psize)
        xst,yst,zst=[np.random.randint(i) for i in st_range]
        x_patch=x[xst:psize[0],yst:psize[1],zst:psize[2]]
        y_patch=y[xst:psize[0],yst:psize[1],zst:psize[2]]
        return x_patch,y_patch
    return patch_extraction

def Rotation3D(max_rate=np.pi/2):
    """
    Rotate a 3D image with alfa, beta and gamma degree respect the axis x, y and z respectively.
    The three angles are chosen randomly between 0-90 degrees
    """
    def rotation3D(x, y):

        alpha, beta, gamma = max_rate*np.random.random_sample(3,)
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(alpha), -np.sin(alpha)],
                    [0, np.sin(alpha), np.cos(alpha)]])
        
        Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]])
        
        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]])
        
        R = np.dot(np.dot(Rx, Ry), Rz)
        
        x_rot = affine_transform(x, R, offset=0, order=3, mode='constant')
        y_rot = affine_transform(y, R, offset=0, order=0, mode='constant')
        
        return x_rot, y_rot
    return rotation3D

def Flip3D():
    """
    Flip the 3D image randomly
    """
    def flip3D(x, y):
        choice = np.random.randint(2,size=(3))
        choice=choice*2-1

        x_flip = x[::choice[0], ::choice[1], ::choice[2]]
        y_flip = y[::choice[0], ::choice[1], ::choice[2]]
        
        return x_flip, y_flip
    return flip3D

def Elastic(sigma=2,order=[1,0]):
    """
    随机弹性形变
    """
    def elastic(x,y):
        xel, yel = elasticdeform.deform_random_grid([x, y], sigma=sigma, axis=[(0, 1, 2), (0, 1, 2)], order=order, mode='constant')
        return xel,yel
    return elastic

def Brightness(down=0.8,up=1.2):
    """
    Changing the brighness of a image using power-law gamma transformation.
    Gain and gamma are chosen randomly for each image channel.

    Gain chosen between [0.8 - 1.2]
    Gamma chosen between [0.8 - 1.2]

    new_im = gain * im^gamma
    """
    def brightness(x, y):
       
        x_new = np.zeros(x.shape)
        gain, gamma = (up - down) * np.random.random_sample(2,) + down
        x_new = np.sign(x)*gain*(np.abs(x)**gamma)
        
        return x_new, y
    return brightness

def resize(img,tg_shape)->np.ndarray:
    orisize=np.array(img.shape)
    tgsize=np.array(tg_shape)
    return scipy.ndimage.zoom(img,tgsize/orisize,order=1)

def normalize(img)->np.ndarray:
    brain= img[img!=0]
    brain_norm=(brain-np.mean(brain))/np.std(brain)
    img_norm=np.zeros_like(img)
    img_norm[img!=0]=brain_norm
    return img_norm

def combine_aug(x, y, arg_list):
    """leave 25% data unchanged"""
    x,y=normalize(x),normalize(y)
    if np.random.random_sample()>0.75:
        return x,y
    
    for arg_func in arg_list:
        x,y=arg_func(x,y)
    return x,y

default_argfunc_pool=[
        Flip3D(),
        Brightness(down=0.8,up=1.2),
        # Rotation3D(max_rate=np.pi/6),
        Elastic(sigma=2,order=[1,0]),
    ]
    
# @tf.function()
def random_jitter(input_image,real_image,argfunc_pool=default_argfunc_pool):

    n=len(argfunc_pool)
    decision=np.random.randint(2, size=(n))
    arg_list=[]
    # print(decision)
    for i in range(n):
        if decision[i]:arg_list.append(argfunc_pool[i])
    return combine_aug(input_image, real_image, arg_list)
    
