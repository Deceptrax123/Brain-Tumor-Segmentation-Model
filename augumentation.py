from scipy import ndimage
import tensorflow as tf
import random


@tf.function
def rotate(volume):

    def sci_rotate(volume):
        angles = [-60, -20, -10, -5, 5, 10, 20, 60, 90]

        angle = random.choice(angles)

        volume = ndimage.rotate(volume, angle, reshape=False)
        return volume

    augumented_volume = tf.numpy_function(sci_rotate, [volume], tf.float32)

    return augumented_volume
