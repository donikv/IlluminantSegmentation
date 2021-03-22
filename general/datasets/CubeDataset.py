import tensorflow as tf
import os
import numpy as np
from general.processing import histogram, augmentations, data_processing

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_HEIGHT = 512
IMG_WIDTH = 1024
TRAIN = 'train'
TEST = 'test'
VALID = 'valid'
fax = os.name != 'nt'
SATURATION_POINT = 2**14 - 2
MASK = np.ones((1732, 2601, 3))
MASK[1050:, 2050:] = np.zeros_like(MASK[1050:, 2050:])
MASK = tf.cast(tf.constant(MASK), tf.float32)

def load_image_names(path, base_path):
    pth = os.path.join(base_path, path)
    names = np.loadtxt(pth, dtype="str")
    names = np.array([base_path + n for n in names])
    return names

def regression_dataset(im_names, indices, gts, type=TRAIN, bs=2, cache=True, uv=True, map_fn=lambda *x: x, sz=(IMG_HEIGHT//2, IMG_WIDTH//2), multi_illuminant=False):
    list_ds = tf.data.Dataset.from_tensor_slices(im_names)
    ind_ds = tf.data.Dataset.from_tensor_slices(indices)
    list_ds = list_ds.zip((list_ds, ind_ds))
    gts = tf.convert_to_tensor(gts)
    labeled_ds = list_ds.map(lambda x, y: process_path_regression(x, y, type, uv, gts, map_fn, sz, multi_illuminant), num_parallel_calls=AUTOTUNE)
    train_ds = data_processing.prepare_for_training(labeled_ds, bs=bs, type=type, cache=cache)
    return train_ds

def get_mask(ind, gts):
    gt = gts[ind]
    return gt


def get_image(file_path):
    file = tf.io.read_file(file_path)
    img = decode_img(file)
    img_norm = tf.linalg.norm(img) / 20
    img = img * scaling / img_norm
    img = tf.where(tf.reduce_max(img, axis=-1, keepdims=True) >= 0.99, tf.zeros_like(img), img)
    return img[...,0:3]


def decode_img(img):
    img = tf.image.decode_png(img)
    img = tf.where(tf.reduce_max(img, axis=-1, keepdims=True) < SATURATION_POINT, img, tf.zeros_like(img))
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img * MASK

    return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])


def process_path_regression(file_path, ind, type, uv, gts, map_fn, sz, mi):
    gt = get_mask(ind, gts)
    # load the raw data from the file as a string
    img = get_image(file_path)
    if type == TRAIN:
        img = augmentations.augment(img, mask_image=True)[0]
    img = tf.image.resize(img, sz)

    if uv:
        img, Iy = histogram.to_uv(img)
        img = tf.stack([Iy, img[...,0], img[...,1]], axis=-1)
        gt = tf.reshape(gt, (-1, 3))
        gt, masky = histogram.to_uv(gt)
        gt = tf.reshape(gt, (-1,))

    if mi:
        return map_fn(img, tf.ones_like(img) * 0.5, gt)
    return map_fn(img, gt)
