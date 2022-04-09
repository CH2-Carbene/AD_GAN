from pet_cycgan.prep import random_jitter
import numpy as np
import tensorflow as tf
def load_np_data(filename,argu=False):
    st=(17, 26, 5)
    ed=(-18, -22, -30)
    if type(filename)!=str:
        filename=filename.decode()
    data=np.load(filename,mmap_mode="r")
    
    # if argu:
        # x,y=random_jitter(data)
    # else:
    data=data[:,st[0]:ed[0],st[1]:ed[1],st[2]:ed[2]]
    if argu:
        x,y=random_jitter(data,only_select=True)
    else:
        x,y=random_jitter(data,only_select=False)
    # x,y=np.pad(x,((16,16),(0,0),(16,16))),np.pad(y,((16,16),(0,0),(16,16)))
    mask=(x!=0)&(y!=0)
    x=x*mask;y=y*mask
    # if GAN_DIRECT=="FA-T1":x,y=y,x
    return tf.convert_to_tensor(x[...,tf.newaxis]),tf.convert_to_tensor(y[...,tf.newaxis])
