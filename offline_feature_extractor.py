from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

from utils import read_image, folder_to_images
from feature_extractor import feature_extractor

root_images_path = "./images/"
saved_features_path = "./feature/"
categories = ["animal", "country", "furniture", "plant", "scenery"]
img_size = (224, 224)

# fe = feature_extractor()
imgs_feature = []
paths_feature = []

for folder in os.listdir(root_images_path):
    if folder.split("_")[0] in categories:
        path = root_images_path + folder
        images_np, images_path = folder_to_images(path)
        paths_feature.extend(np.array(images_path))
        imgs_feature.extend(feature_extractor(images_np))
        
print(imgs_feature)