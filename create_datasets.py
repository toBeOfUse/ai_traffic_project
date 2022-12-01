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

def get_datasets(
    img_size: int, small_test: bool=False
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Returns two datasets in a tuple: (training, validation)
    """
    print("loading images as dataset...")
    training, validation = tf.keras.utils.image_dataset_from_directory(
        Path("./udacity-2/object-dataset/smaller/").absolute(),
        labels=vectors,
        label_mode="categorical",
        color_mode="rgb",
        image_size=(img_size, img_size),
        validation_split=0.8,
        seed=2**20,
        subset="both",
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
    )
    if small_test:
        return training.take(400), validation.take(100)
    else:
        return training, validation
