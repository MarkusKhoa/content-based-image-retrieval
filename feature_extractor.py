import tensorflow as tf


def feature_extractor(input_img):
    x = tf.keras.applications.vgg19.preprocess_input(input_img)
    base_model = tf.keras.applications.vgg19.VGG19(weights = "imagenet",
                                                   include_top = False)
    
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs = input_img, outputs = x)
    feature = model.predict(x)
    
    return feature