import tensorflow as tf
import numpy as np

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def diceLoss(y_true, y_pred):
    
    y_true = tf.convert_to_tensor(y_true, 'float32')
    y_pred = tf.convert_to_tensor(y_pred[...,0], y_true.dtype)

    num = tf.math.reduce_sum(tf.math.multiply(y_true, y_pred))
    den = tf.math.reduce_sum(tf.math.add(y_true, y_pred))+1e-5

    return 1-2*num/den

def get_L2loss(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, 'float32')
    y_pred = tf.convert_to_tensor(y_pred[...,0], y_true.dtype)

    return tf.math.reduce_mean(tf.math.pow(y_true - y_pred, 2))
    num = tf.math.reduce_sum(tf.math.multiply(y_true, y_pred))
    den = tf.math.reduce_sum(tf.math.add(y_true, y_pred))+1e-5

def get_L1loss(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, 'float32')
    y_pred = tf.convert_to_tensor(y_pred[...,0], y_true.dtype)

    return tf.math.reduce_mean(tf.math.abs(y_true - y_pred))

def get_disc_loss(disc_real_output, disc_fake_output):
    # real_loss = tf.math.reduce_mean(tf.math.pow(tf.ones_like(disc_real_output) - disc_real_output, 2))
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    # fake_loss = tf.math.reduce_mean(tf.math.pow(tf.zeros_like(disc_fake_output) - disc_fake_output, 2))
    fake_loss = loss_object(tf.zeros_like(disc_fake_output), disc_fake_output)

    disc_loss = 0.5*(real_loss + fake_loss)

    return disc_loss


def get_gen_loss(target, gen_output, disc_fake_output, alpha):

    # generalized dice loss
    # dice_loss = diceLoss(target, gen_output)
    L1_loss=get_L1loss(target,gen_output )#tf.math.reduce_mean(tf.math.pow(gen_output - target, 2))
    # disc loss
    # disc_loss = tf.math.reduce_mean(tf.math.pow(tf.ones_like(disc_fake_output) - disc_fake_output, 2))
    # disc_loss = tf.math.reduce_mean(tf.math.pow(tf.ones_like(disc_fake_output) - disc_fake_output, 2))
    disc_loss = loss_object(tf.ones_like(disc_fake_output), disc_fake_output)
    # total loss
    gen_loss = alpha*L1_loss + disc_loss

    return gen_loss, L1_loss, disc_loss
