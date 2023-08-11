import tensorflow as tf


def scheduler(epoch, lr):
    initial = 0.001
    if (epoch <= 30):
        return initial  # warmup phase
    else:
        return lr*tf.math.exp(-0.1*(epoch-30))  # decay phase
