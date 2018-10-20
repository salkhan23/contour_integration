# ------------------------------------------------------------------------------------------------
# Collect a random subset of images and labels to use to test one pixel adverserial attacks
# ------------------------------------------------------------------------------------------------
import numpy as np
import os
from shutil import copyfile
import pickle

IMAGE_NET_TRAIN_DIR = '/media/salman/076d0e17-1483-4b67-ba60-aa8e7efc8edf/SalmanExternal/' \
                     'ImageNet_ILSVRC2012/imagenet-data/train'
IMAGE_NET_VALIDATION_DIR = '/media/salman/076d0e17-1483-4b67-ba60-aa8e7efc8edf/SalmanExternal/' \
                     'ImageNet_ILSVRC2012/imagenet-data/validation'

IMAGE_NET_META_DATA_FILE = '/media/salman/076d0e17-1483-4b67-ba60-aa8e7efc8edf/SalmanExternal/' \
    '/ImageNet_ILSVRC2012/processing/imagenet_metadata.txt'

STORE_IMG_DIR = '/media/salman/076d0e17-1483-4b67-ba60-aa8e7efc8edf/SalmanExternal/' \
    '/ImageNet_ILSVRC2012/sample_images_5'

if not os.path.exists(STORE_IMG_DIR):
    os.mkdir(STORE_IMG_DIR)

if __name__ == '__main__':

    np.random.seed(57)

    # Label 2 Words Converter
    with open(IMAGE_NET_META_DATA_FILE, 'rb') as handle:
        z = handle.readlines()

    label_2_words = {}
    for item in z:
        label, word = item.strip().split('\t', 1)
        label_2_words[label] = word


    # Get images
    n_images_per_cat = 1

    list_of_dirs = os.listdir(IMAGE_NET_VALIDATION_DIR)
    #list_of_dirs = list_of_dirs[100:700]

    n_dir = len(list_of_dirs)

    stored_images = {}

    for d_idx, d in enumerate(list_of_dirs):
        for img_idx in np.arange(n_images_per_cat):

            # Randomize
            img_idx = np.random.randint(0, 50, size=1)
            img_idx = np.int(img_idx)

            img = os.listdir(os.path.join(IMAGE_NET_VALIDATION_DIR, list_of_dirs[d_idx]))[img_idx]

            label = list_of_dirs[d_idx]

            copyfile(
                os.path.join(IMAGE_NET_VALIDATION_DIR, d, img),
                os.path.join(STORE_IMG_DIR, img))

            stored_images[img] = (label, label_2_words[label])

    for k,v in stored_images.iteritems():
        print k, v

    data_key_file = os.path.join(STORE_IMG_DIR, 'data_key.pickle')

    with open(data_key_file, 'wb') as handle:
        pickle.dump(stored_images, handle)
