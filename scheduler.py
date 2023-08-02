import tensorflow as tf


def scheduler(epoch, lr):
    if (epoch <= 25):
        return epoch*0.00008+0.001  # warmup phase
    else:
        return lr*tf.math.exp(-0.1*epoch)  # decay phase
