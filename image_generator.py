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
            angles = [0, 30, 45, 60, 90, 120, 135]
            angle = random.choice(angles)

            sample = ndimage.rotate(sample, angle, reshape=False)
            mask = ndimage.rotate(mask, angle, reshape=False)

        # Normalization
        for i in range(3):
            sample[:, :, :, i] = (
                sample[:, :, :, i]-np.mean(sample[:, :, :, i], axis=0))/np.std(sample[:, :, :, i], axis=0)
            sample[:, :, :, i] = np.round(sample[:, :, :, i], 6)

        yield sample, mask
