import tensorflow as tf
import numpy as np
def diceLoss(y_true, y_pred):
    
    y_true = tf.convert_to_tensor(y_true, 'float32')
    y_pred = tf.convert_to_tensor(y_pred[...,0], y_true.dtype)

    num = tf.math.reduce_sum(tf.math.multiply(y_true, y_pred))
    den = tf.math.reduce_sum(tf.math.add(y_true, y_pred))+1e-5

    return 1-2*num/den

def get_disc_loss(disc_real_output, disc_fake_output):
    real_loss = tf.math.reduce_mean(tf.math.pow(tf.ones_like(disc_real_output) - disc_real_output, 2))
    fake_loss = tf.math.reduce_mean(tf.math.pow(tf.zeros_like(disc_fake_output) - disc_fake_output, 2))

    disc_loss = 0.5*(real_loss + fake_loss)

    return disc_loss


def get_gen_loss(target, gen_output, disc_fake_output, alpha):

    # generalized dice loss
    dice_loss = diceLoss(target, gen_output)

    # disc loss
    disc_loss = tf.math.reduce_mean(tf.math.pow(tf.ones_like(disc_fake_output) - disc_fake_output, 2))

    # total loss
    gen_loss = alpha*dice_loss + disc_loss

    return gen_loss, dice_loss, disc_loss
