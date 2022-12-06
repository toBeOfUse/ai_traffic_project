import tensorflow as tf
from create_model import traffic_lights_fn
from create_datasets import labels, file_to_vector
import numpy as np
import random

trained_model = tf.keras.models.load_model('saved_model/traffic_lights', custom_objects={"traffic_lights_fn": traffic_lights_fn})

# Check its architecture
trained_model.summary()

def predict(image_path: str):
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224,224),
        interpolation="bilinear"
    )
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = trained_model.predict(input_arr)
    print(predictions)
    for result in predictions:
        for i in range(len(result)):
            if result[i] > 0.5:
                print("image has", labels[i])
    return predictions

if __name__ == "__main__":
    path = "udacity-2/object-dataset/"
    files = list(file_to_vector.keys())
    tlgi = labels.index("trafficLightGreen")
    tlri = labels.index("trafficLightRed")
    correct = 0
    for i in range(100):
        filename = random.choice(files)
        prediction = [round(x) for x in predict(path+filename)]
        if prediction[tlri] == files[filename][tlri]:
            correct += 1
    print("detected", correct, "red lights out of 100")
