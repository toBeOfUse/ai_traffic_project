import tensorflow as tf
import tensorflow_hub as hub
from create_datasets import get_datasets, labels

def traffic_lights_fn(y_true: tf.Tensor, y_pred: tf.Tensor):
    """Traffic light false negative rate - percentage of inputs that had traffic
    lights but the traffic light was not detected"""
    tl_index = labels.index("trafficLight")
    light_prediction = tf.cast(
        tf.math.round(tf.map_fn(lambda x: x[tl_index], y_pred)),
        tf.bool
    )
    light_actual = tf.cast(
        tf.map_fn(lambda x: x[tl_index], y_true),
        tf.bool
    )
    return tf.cast(
        tf.math.logical_and(light_actual, tf.math.logical_not(light_prediction)),
        tf.float32
    )


def get_model():
    # get datasets with image size (224, 224) to match imported mobilenet model
    img_size = 224
    training, validation = get_datasets(img_size)

    layers = tf.keras.layers

    # transfer learning
    feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5"
    feature_extractor_layer = hub.KerasLayer(
        feature_extractor_url, input_shape=(img_size, img_size, 3), trainable=False
    )

    model = tf.keras.Sequential([
        feature_extractor_layer,
        layers.Dense(1024, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(len(labels), activation='sigmoid')
    ])
    model.summary()

    # test predicting output with no training to make sure layers are in place
    model.predict(next(iter(training))[0])[:1]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss=tf.keras.metrics.binary_crossentropy,
        metrics=["binary_crossentropy"]
    )

    history = model.fit(training, epochs=5, validation_data=validation)

    model.save("saved_model/traffic_lights")

if __name__ == "__main__":
    get_model()
