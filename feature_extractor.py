import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from utils import folder_to_images

class FeatureExtractor:
    def __init__(self):
        base_model = ResNet50(weights = "imagenet", include_top = False)
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        self.model = tf.keras.Model(inputs = base_model.input, outputs = x)
    def call(self, input_img):
        x = preprocess_input(input_img)
        feature = self.model.predict(x)
        return feature

imgs_feature = []
paths_feature = []
FE = FeatureExtractor()

root_images_path = "./images/"
saved_features_path = "./feature/"
os.makedirs(saved_features_path, exist_ok = True)
categories = ["animal", "country", "furniture", "plant", "scenery"]
img_size = (224, 224)