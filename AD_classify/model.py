import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Conv3D, Conv3DTranspose, Dropout, ReLU, LeakyReLU, Concatenate, ZeroPadding3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization

K_INITER = "he_normal"

def Classify():
    '''
    Classify model between NC/MCI
    '''
    
    mod=2
    Nfilter_start = 16
    depth = 3
    ks = 4
    embedding_channel=4
    inputs = [Input((64,64,64,1), name=f'input_image_{i}') for i in range(mod)]
    # inputs = [Input((128,128,128,1), name=f'input_image_0')]

    def encoder_step(layer, Nf, norm=True):
        x = Conv3D(Nf, kernel_size=ks, strides=2, kernel_initializer=K_INITER, padding='same')(layer)
        if norm:
            x = InstanceNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        
        return x

    x_list = [x for x in inputs]
    for i in range(len(x_list)):
        x=x_list[i]
        for d in range(depth):
            if d==0:
                x = encoder_step(x, Nfilter_start*np.power(2,d), False)
            else:
                x = encoder_step(x, Nfilter_start*np.power(2,d))

        x = ZeroPadding3D()(x)
        x = Conv3D(Nfilter_start*(2**depth), ks, strides=1, padding='valid', kernel_initializer=K_INITER)(x) 
        x = InstanceNormalization()(x)
        x = LeakyReLU()(x)
        # x = Conv3D(embedding_channel, ks, strides=1, padding='valid', kernel_initializer=K_INITER)(x) 
        # x = LeakyReLU()(x)
        
        x_list[i]=x

    x = Concatenate()(x_list)
    
    x = ZeroPadding3D()(x)
    last = Conv3D(1, ks, strides=1, padding='valid', kernel_initializer=K_INITER, name='output_classify')(x)

    return Model(inputs=inputs, outputs=last, name='Classify')

if __name__ == '__main__':
    m=Classify()
    m.summary(line_length=120)