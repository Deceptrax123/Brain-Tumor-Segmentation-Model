import numpy as np
import random
from scipy import ndimage
import glob
import os


def get_id(path):
    slice = path.split(".")[1]
    id = slice[len(slice)-1:len(slice)-4:-1]

    return id[-1::-1]


def generate_image(paths, train):
    for img_path in paths:
        example_id = get_id(img_path)

        sample = np.load(img_path)

        mask = np.load("./Data/Train/masks_reformatted/mask_" +
                       example_id+".npy")

        if (train == True):
            angles = [-60, -20, -10, -5, 0, 5, 10, 20, 60, 90]
            angle = random.choice(angles)
            sample = ndimage.rotate(sample, angle, reshape=False)

        # Normalization
        normalized_sample = (sample-np.mean(sample, axis=0)
                             )/np.std(sample, axis=0)

        yield normalized_sample, mask
