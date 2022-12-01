from pathlib import Path
print("importing tensorflow...")
import tensorflow as tf

print("gathering label vectors...")
vectors = []
labels = []
with open("category_vector.csv") as label_file:
    labels = label_file.readline().strip().split(",")[1:]
    for label_line in label_file:
        vectors.append(list(map(int, label_line.split(",")[1:])))
print("labels:", labels)
print("first label vector:", vectors[0])

print("loading images as dataset...")
training, validation = tf.keras.utils.image_dataset_from_directory(
    Path("./udacity-2/object-dataset/smaller/").absolute(),
    labels=vectors,
    label_mode="categorical",
    color_mode="rgb",
    image_size=(256, 256),
    validation_split=0.8,
    seed=2**20,
    subset="both",
    interpolation="bilinear",
    crop_to_aspect_ratio=False,
)

print(training)
print(validation)
