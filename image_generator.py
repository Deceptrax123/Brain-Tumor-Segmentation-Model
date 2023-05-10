import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn.decomposition import PCA, IncrementalPCA
import glob
import os


def get_id(path):
    slice = path.split(".")[1]
    id = slice[len(slice)-1:len(slice)-4:-1]

    return id[-1::-1]


def generate_image(paths):
    for img_path in paths:
        example_id = get_id(img_path)

        sample = np.load(img_path)

        mask = np.load("./Data/Train/masks_reformatted/mask_" +
                       example_id+".npy")
        # reshape mask
        # mask_reshaped=np.reshape(mask,(128*128*128,4))
        # Normalization
        normalized_sample = (sample-np.mean(sample, axis=0)
                             )/np.std(sample, axis=0)

        yield normalized_sample, mask
