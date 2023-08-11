import numpy as np
import tensorflow as tf
from image_generator import generate_image
from augumentation import rotate

# test generator


def test_batch_generator(paths, batch_size):
    while True:
        image = generate_image(paths)

        img_batch = list()
        mask_batch = list()

        for sample, mask in image:
            img_batch.append(sample)
            mask_batch.append(mask)

            if (len(img_batch) == batch_size):
                yield np.stack(img_batch, axis=0), np.stack(mask_batch, axis=0)
                img_batch = []
                mask_batch = []

        if (len(img_batch) != 0):
            yield np.stack(img_batch, axis=0), np.stack(mask_batch, axis=0)
            img_batch = []
            mask_batch = []


def train_batch_generator(paths, batch_size):
    while True:
        image = generate_image(paths)

        # rotate the images by a random choice of angles
        image = rotate(image)

        img_batch = list()
        mask_batch = list()

        for sample, mask in image:
            img_batch.append(sample)
            mask_batch.append(mask)

            if (len(img_batch) == batch_size):
                yield np.stack(img_batch, axis=0), np.stack(mask_batch, axis=0)
                img_batch = []
                mask_batch = []

        if (len(img_batch) != 0):
            yield np.stack(img_batch, axis=0), np.stack(mask_batch, axis=0)
            img_batch = []
            mask_batch = []
