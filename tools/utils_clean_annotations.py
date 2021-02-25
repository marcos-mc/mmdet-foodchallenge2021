import cv2
import json
import os

from tqdm import tqdm


def fix_data(annotations, directory):
    for n, i in enumerate(tqdm((annotations['images']))):

        img = cv2.imread(directory + '/' + i["file_name"])

        if img.shape[0] != i['height']:
            annotations['images'][n]['height'] = img.shape[0]

        if img.shape[1] != i['width']:
            annotations['images'][n]['width'] = img.shape[1]

    return annotations


dataset_path = r'/media/HDD_4TB_1/Datasets/AICrowd_newval'
train_annotations_path = os.path.join(dataset_path, 'train', 'new_annotations_mapped.json')
train_images_path = os.path.join(dataset_path, 'train', 'images')
val_annotations_path = os.path.join(dataset_path, 'val', 'new_annotations_mapped.json')
val_images_path = os.path.join(dataset_path, 'val', 'images')

with open(train_annotations_path) as f:
    train_annotations_data = json.load(f)
    fixed_annotations_data = fix_data(train_annotations_data, train_images_path)
    with open(os.path.join(dataset_path, 'train', 'train_annotations_fixed.json'), 'w') as f:
        json.dump(fixed_annotations_data, f)

with open(val_annotations_path) as f:
    val_annotations_data = json.load(f)
    fixed_annotations_data = fix_data(val_annotations_data, val_images_path)
    with open(os.path.join(dataset_path, 'val', 'val_annotations_fixed.json'), 'w') as f:
        json.dump(fixed_annotations_data, f)

