from collections import defaultdict
from pathlib import Path

all_labels = set()
file_labels = defaultdict(list)
with open("./udacity-2/object-dataset/labels.csv") as label_file:
    for label in label_file:
        comps = label.split()
        image_file = comps[0]
        label = comps[6].strip('"')
        file_labels[image_file].append(label)
        all_labels.add(label)
        if label == "trafficLight" and len(comps) == 8:
            sublabel = "trafficLight"+comps[7].strip('"')
            file_labels[image_file].append(sublabel)
            all_labels.add(sublabel)

labels_list = sorted(all_labels)
file_category_vectors = {}

for file in Path("./udacity-2/object-dataset/smaller/").glob("*.jpg"):
    labels = file_labels[file.name]
    file_category_vectors[file.name] = [0]*len(labels_list)
    for label in labels:
        file_category_vectors[file.name][labels_list.index(label)] = 1
file_vectors_list = list(file_category_vectors.items())
file_vectors_list.sort(key=lambda x: x[0])

with open("category_vector.csv", mode="w+") as outfile:
    outfile.write("file_name,"+",".join(labels_list)+"\n")
    for file, vector in file_vectors_list:
        outfile.write(",".join([file, *map(str, vector)]) + "\n")
