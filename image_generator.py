import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img


def get_id(path):
    slice = path.split(".")[1]
    id = slice[len(slice)-1:len(slice)-4:-1]

    return id


def generate_image(paths, size=(128, 128, 128)):
    for path in paths:
        example_id = get_id(path)

        sample = np.load(path)
        mask = np.load("./Data/Train/masks_reformatted/mask_" +
                       example_id+".npy")

        yield sample, mask
