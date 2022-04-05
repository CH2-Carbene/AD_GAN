import tensorflow as tf
import numpy as np


def classify_loss(label, classify_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # real_loss = tf.math.reduce_mean(tf.math.pow(tf.ones_like(disc_real_output) - disc_real_output, 2))
    return loss_object(tf.ones_like(classify_output)*label, classify_output)

    # fake_loss = tf.math.reduce_mean(tf.math.pow(tf.zeros_like(disc_fake_output) - disc_fake_output, 2))

