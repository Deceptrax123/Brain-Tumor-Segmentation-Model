import tensorflow as tf

def scheduler(epoch,lr):
    return lr*tf.math.exp(-0.1*epoch)