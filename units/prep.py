from scipy.ndimage.interpolation import affine_transform
import elasticdeform,scipy
import numpy as np
from units.base import fn_pipe
def Patch_extration(psize=(128,128,128),sp=None):
    def patch_extraction(x,y):
        """
        3D patch extraction
        """
        # xlen,ylen,zlen=x.shape
        st_range=np.array(x.shape)-np.array(psize)
        
        if sp is None:
            start_pos=[np.random.randint(i) for i in st_range]
        else:
            start_pos=sp
        xst,yst,zst=start_pos
        xed,yed,zed=[st+sz for st,sz in zip([xst,yst,zst],psize)]
        x_patch=x[xst:xed,yst:yed,zst:zed]
        y_patch=y[xst:xed,yst:yed,zst:zed]
        return x_patch,y_patch
    return patch_extraction

def Rotation3D(max_rate=np.pi/2):
    """
    Rotate a 3D image with alfa, beta and gamma degree respect the axis x, y and z respectively.
    The three angles are chosen randomly between 0-90 degrees
    """
    def rotation3D(data):

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
        for i in range(len(data)):
            data[i]=affine_transform(data[i], R, offset=0, order=3, mode='constant')
        # x_rot = 
        # y_rot = affine_transform(y, R, offset=0, order=0, mode='constant')
        
        return data
    return rotation3D

def Flip3D(axis=(1,0,0)):
    """
    Flip the 3D image randomly
    """
    choice = np.random.randint(2,size=(3))*np.array(axis)
    choice=1-choice*2
    def flip3D(data):


        data = data[:,::choice[0], ::choice[1], ::choice[2]]
        # y_flip = y[::choice[0], ::choice[1], ::choice[2]]
        
        return data
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
    def brightness(data):

        x=data[0]
        x_new = np.zeros(x.shape)
        gain, gamma = (up - down) * np.random.random_sample(2,) + down
        x_new = np.sign(x)*gain*(np.abs(x)**gamma)
        data[0]=x_new

        return data
    return brightness

def Static_select(tgsize=(128,128,128)):
    select_size=np.array(tgsize)
    
    def static_select(data):
        # print(data.shape)
        st_range=np.array(data[0].shape)-select_size
        st=np.random.randint(st_range+1)
        ed=st+select_size
        data=data[:,st[0]:ed[0],st[1]:ed[1],st[2]:ed[2]]
        # mask=(x!=0)&(y!=0)
        # x,y=x*mask,y*mask
        return data
    return static_select

def Random_select(tgsize=(128,128,128),low=0.8,high=1.2):
    tgsize_arr=np.array(tgsize)
    select_size=(np.random.uniform(low,high,size=(3))*tgsize_arr).astype(int)

    def random_select(data):
        # print(data.shape)
        # print(data[0].shape)
        st_range=np.array(data[0].shape)-select_size
        pw=(st_range<0)*(-st_range)
        pw=((0,0),)+tuple(((pw[i]+1)//2,(pw[i]+1)//2) for i in range(len(pw)))
        # print("PW=",pw)
        data=np.pad(data,pad_width=pw)
        st_range=np.array(data[0].shape)-select_size
        
        st=np.random.randint(st_range+1)
        ed=st+tgsize_arr
        data=data[:,st[0]:ed[0],st[1]:ed[1],st[2]:ed[2]]
        resized_data=[resize(img,tgsize)for img in data]
        # mask=(x!=0)&(y!=0)
        # x,y=x*mask,y*mask
        # return data
        return np.array(resized_data)
    return random_select

def resize(img,tg_shape)->np.ndarray:
    orisize=np.array(img.shape)
    tgsize=np.array(tg_shape)
    return scipy.ndimage.zoom(img,tgsize/orisize,order=1)

def normalize(img:np.ndarray,save_rate=0.99)->np.ndarray:
    brain= img[img!=0]
    n=len(brain)
    maxp,minp=round(n*((1+save_rate)/2)),round(n*((1-save_rate)/2))
    
    maxn,minn=np.partition(brain,kth=maxp)[maxp],np.partition(brain,kth=minp)[minp]
    brain[brain>maxn]=maxn
    # brain[brain<minn]=minn
    brain_norm=(brain)/maxn
    
    img_norm=np.zeros_like(img)
    img_norm[img!=0]=brain_norm
    return img_norm

# def combine_aug(data, arg_list):
#     """leave 25% data unchanged"""
#     # x,y=normalize(x),normalize(y)

#     for arg_func in arg_list:
#         x,y=arg_func(x,y)
#     return x,y

default_argfunc_pool=[
    Flip3D(axis=[1,0,0]),
    Brightness(down=0.8,up=1.2),
    Rotation3D(max_rate=np.pi/18),#考虑减小到10°，消融对比
    # Elastic(sigma=2,order=[1,0]),
]

# @tf.function()
def random_jitter(input_data,argfunc_pool=default_argfunc_pool,only_select=False,select_shape=(128,128,128)):

    n=len(argfunc_pool)
    decision=np.random.randint(2, size=(n))
    arg_list=[]
    # print(decision)
    if only_select or np.random.random_sample()<0.2:
        arg_list.append(Static_select(select_shape))
    else:
        arg_list.append(Random_select(select_shape)) #Select is must
        for i in range(n):
            if decision[i]:arg_list.append(argfunc_pool[i])
        
    combine_arg=fn_pipe(arg_list)
    data=combine_arg(input_data)
    # combine_aug((input_data), arg_list)
    return data