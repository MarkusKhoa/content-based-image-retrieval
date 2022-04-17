from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

def read_image(path, img_size):
    img = Image.open(path).resize(img_size)
    return np.asarray(img, dtype=np.float32)

