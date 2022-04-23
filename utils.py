from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

def read_image(path, img_size):
    """
    args:
        path (path of the input image file): string
        img_size (size of the input image file): tuple
    return:
        numpy array of the input image
    """
    img = Image.open(path).resize(img_size)
    return np.asarray(img, dtype=np.float32)

def folder_to_images(folder, img_size = (224, 224)):
    """
    args:
        folder (folder name of root images paths): string
        img_size: default to (224, 224): tuple
    
    return:
        numpy arrays of images and images' files in the given folder
    """
    list_dir = [folder + '/' + name for name in os.listdir(folder) if name.endswith((".jpg", ".png", ".jpeg"))]
    i = 0
    imgs_np = np.zeros(shape=(len(list_dir), *img_size, 3))
    imgs_list = []
    for img_file in list_dir:
        try:
            imgs_np[i] = read_image(img_file, img_size)
            imgs_list.append(img_file)
            i += 1
        except Exception:
            print(f"Cannot read image: {img_file}")
            # os.remove(img_path)
    imgs_list = np.array(imgs_list)
    return imgs_np, imgs_list