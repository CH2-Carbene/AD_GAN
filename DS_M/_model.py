import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Conv3D, Conv3DTranspose, Dropout, ReLU, LeakyReLU, Concatenate, ZeroPadding3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization

K_INITER = "he_normal"

def Generator():
    def encoder_step(layer, Nf, ks, norm=True):
        x = Conv3D(Nf, kernel_size=ks, strides=2, kernel_initializer=K_INITER, padding='same')(layer)
        if norm:
            x = InstanceNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)

        return x

    def bottlenek(layer, Nf, ks):
        x = Conv3D(Nf, kernel_size=ks, strides=2, kernel_initializer=K_INITER, padding='same')(layer)
        x = InstanceNormalization()(x)
        x = LeakyReLU()(x)
        for i in range(4):
            y = Conv3D(Nf, kernel_size=ks, strides=1, kernel_initializer=K_INITER, padding='same')(x)
            x = InstanceNormalization()(y)
            x = LeakyReLU()(x)
            x = Concatenate()([x, y])

        return x

    def decoder_step(layer, layer_to_concatenate, Nf, ks):
        x = Conv3DTranspose(Nf, kernel_size=ks, strides=2, padding='same', kernel_initializer=K_INITER)(layer)
        x = InstanceNormalization()(x)
        x = LeakyReLU()(x)
        x = Concatenate()([x, layer_to_concatenate])
        x = Dropout(0.2)(x)
        return x
    
    layers_to_concatenate = []
    inputs = Input((128,128,128,1), name='input_image')
    Nfilter_start = 32
    depth = 4
    ks = 4
    x = inputs
    
    # encoder
    for d in range(depth-1):
        if d==0:
            x = encoder_step(x, Nfilter_start*np.power(2,d), ks, False)
        else:
            x = encoder_step(x, Nfilter_start*np.power(2,d), ks)
        layers_to_concatenate.append(x)

    # bottlenek
    x = bottlenek(x, Nfilter_start*np.power(2,depth-1), ks)
    
    # decoder
    for d in range(depth-2, -1, -1): 
        x = decoder_step(x, layers_to_concatenate.pop(), Nfilter_start*np.power(2,d), ks)

    
    # classifier
    last = Conv3DTranspose(1, kernel_size=ks, strides=2, padding='same', kernel_initializer=K_INITER, name='output_generator')(x)
    # last_norm=ReLU()(last/2+1)-1
    return Model(inputs=inputs, outputs=last, name='Generator')

def Discriminator():
    '''
    Discriminator model
    '''

    inputs = Input((128,128,128,1), name='input_image')
    targets = Input((128,128,128,1), name='target_image')
    Nfilter_start = 16
    depth = 3
    default_ks = 4

    def encoder_step(layer, Nf, ks=default_ks, norm=True):
        x = Conv3D(Nf, kernel_size=ks, strides=2, kernel_initializer=K_INITER, padding='same')(layer)
        if norm:
            x = InstanceNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        
        return x

    x = Concatenate()([inputs, targets])

    for d in range(depth):
        if d==0:
            x = encoder_step(x, Nfilter_start*np.power(2,d), default_ks, norm=False)
        else:
            x = encoder_step(x, Nfilter_start*np.power(2,d), default_ks)

    x = ZeroPadding3D()(x)
    x = Conv3D(Nfilter_start*(2**depth), default_ks, strides=1, padding='valid', kernel_initializer=K_INITER)(x) 
    x = InstanceNormalization()(x)
    x = LeakyReLU()(x)
      
    x = ZeroPadding3D()(x)
    last = Conv3D(1, default_ks, strides=1, padding='valid', kernel_initializer=K_INITER, name='output_discriminator')(x)

    return Model(inputs=[targets, inputs], outputs=last, name='Discriminator')

def ensembler():

    start = Input((128,128,128,40))
    fin = Conv3D(4, kernel_size=3, kernel_initializer=K_INITER, padding='same', activation='softmax')(start)

    return Model(inputs=start, outputs=fin, name='Ensembler')


if __name__ == '__main__':
    G,D=Generator(),Discriminator()
    G.summary(line_length=120)
    D.summary(line_length=120)