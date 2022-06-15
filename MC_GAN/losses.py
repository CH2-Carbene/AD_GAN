import tensorflow as tf
import numpy as np
import tensorflow.keras.losses

def L1_loss(x:tf.Tensor,y:tf.Tensor):
    if isinstance(x,tf.Tensor) and isinstance(y,tf.Tensor):
        # x=tf.convert_to_tensor(x, 'float32')
        # y=tf.convert_to_tensor(y, 'float32')
        y=tf.reshape(y,x.shape)
    return tf.reduce_mean(tf.abs(x-y))

def L2_loss(x:tf.Tensor,y:tf.Tensor):
    if isinstance(x,tf.Tensor) and isinstance(y,tf.Tensor):
        # x=tf.convert_to_tensor(x, 'float32')
        # y=tf.convert_to_tensor(y, 'float32')
        y=tf.reshape(y,x.shape)
    return tf.reduce_mean(tf.math.squared_difference(x,y))


if __name__=='__main__':
    a=tf.constant([[1,1,4],[5,1,4]],dtype='float32')
    b=a*2
    b=tf.reshape(b,(2,1,3))
    # b=b[...,tf.newaxis()]
    print(L1_loss(a,1))
    print(L2_loss(a,b))