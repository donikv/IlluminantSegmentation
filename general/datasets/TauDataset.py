import tensorflow as tf
import os
from general.processing import histogram, augmentations
import tensorflow_io as tfio

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_HEIGHT = 512
IMG_WIDTH = 1024
TRAIN = 'train'
TEST = 'test'
VALID = 'valid'
fax = os.name != 'nt'

def regression_dataset(im_names, type=TRAIN, sz=(IMG_HEIGHT//2, IMG_WIDTH//2), bs=2, cache=True, uv=True, map_fn=lambda *x: x):
    list_ds = tf.data.Dataset.from_tensor_slices(im_names)
    labeled_ds = list_ds.map(lambda x: process_path_regression(x, type, uv, map_fn, sz), num_parallel_calls=AUTOTUNE)
    train_ds = prepare_for_training(labeled_ds, bs=bs, type=type, cache=cache)
    return train_ds

def get_mask(file_path):
    path = tf.strings.reduce_join([file_path, 'wp'], separator='.')
    file = tf.io.read_file(path)
    mask = tf.io.decode_csv(file, [1., 1., 1.], field_delim=',')
    return tf.convert_to_tensor(mask)


def get_image(file_path):
    path = tf.strings.reduce_join([file_path, 'tiff'], separator='.')
    file = tf.io.read_file(path)
    img = decode_img(file)
    print(img.shape)
    return img[...,0:3]


def decode_img(img):
    img = tfio.experimental.image.decode_tiff(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])


def process_path_regression(file_path, type, uv, map_fn, sz):
    mask = get_mask(file_path)
    print(mask.shape)
    # load the raw data from the file as a string
    img = get_image(file_path)
    if type == TRAIN:
        img = augmentations.augment(img)[0]
    img = tf.image.resize(img, sz)

    if uv:
        img, Iy = histogram.to_uv(img)
        img = tf.stack([Iy, img[...,0], img[...,1]], axis=-1)
        mask = tf.reshape(mask, (-1, 3))
        mask, masky = histogram.to_uv(mask)
        mask = tf.reshape(mask, (-1,))

    return map_fn(img, mask)


def prepare_for_training(ds, type, cache=True, shuffle_buffer_size=100, bs=2):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    if type == TRAIN:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    if type != TEST:
        # Repeat forever
        ds = ds.repeat()

    ds = ds.batch(bs)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds
