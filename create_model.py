import tensorflow as tf
import tensorflow_hub as hub
from create_datasets import get_datasets, labels

def get_model():
    # get datasets with image size (224, 224) to match imported mobilenet model
    img_size = 224
    training, validation = get_datasets(img_size, True)

    layers = tf.keras.layers

    feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
    feature_extractor_layer = hub.KerasLayer(
        feature_extractor_url, input_shape=(img_size, img_size, 3)
    )
    feature_extractor_layer.trainable = False

    model = tf.keras.Sequential([
        feature_extractor_layer,
        layers.Dense(512, activation='relu'),
        layers.Dense(len(labels), activation='sigmoid')
    ])

    # test predicting output with no training to make sure layers are in place
    model.predict(next(iter(training))[0])[:1]

    return model

if __name__ == "__main__":
    get_model().summary()
