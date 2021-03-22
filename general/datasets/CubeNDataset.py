import tensorflow as tf
import os
import numpy as np
from general.processing import augmentations, data_processing as dp
from general.utils import visualizer

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_HEIGHT = 1024
IMG_WIDTH = 2048
fax = os.name != 'nt'


def load_paths(root, list_name):
    paths = np.loadtxt(os.path.join(root, list_name), dtype=str)
    paths = np.array(list(map(lambda x: os.path.join(root, x[2:]), paths)))
    return paths


def timestamped_histogram_dataset(dataset, p, depth, bs):
    mask_shape = (2, IMG_HEIGHT // 2, IMG_WIDTH // 2, 3)
    timestamps = IMG_HEIGHT // 2 * IMG_WIDTH // 2 // p ** 2
    ds = dp.reduce_dataset(dp.histogram_dataset(dp.patch_dataset(dataset, p, mask_shape, mask_shape), depth), None)
    ds = dp.timestamp_dataset(ds, timestamps, (timestamps * bs, depth**2), (timestamps * bs, 1))
    return ds


def dataset(im_paths, type=dp.TRAIN, bs=2, n=3, cache=True, regression=False, gt=False, round=True, gamma=False, h=IMG_HEIGHT, w=IMG_WIDTH, map_fn=lambda *x: x, shuffle_buffer_size=100):
    scaling = tf.data.Dataset.from_tensor_slices([1 if x.find('canon_550d/outdoor2/') != -1 or x.find('canon_550d/lab6/') != -1 else 1/4 for x in im_paths]) if gamma else tf.data.Dataset.from_tensor_slices([4 if x.find('canon_550d/outdoor2/') != -1 or x.find('canon_550d/lab6/') != -1 else 1 for x in im_paths])
    list_ds = tf.data.Dataset.from_tensor_slices(im_paths)
    list_ds = list_ds.zip((list_ds, scaling))
    labeled_ds = list_ds.map(lambda x,s: process_path(x, s, type, regression, h, w, n, gt, round, gamma, map_fn), num_parallel_calls=AUTOTUNE)
    train_ds = dp.prepare_for_training(labeled_ds, bs=bs, type=type, cache=cache, shuffle_buffer_size=shuffle_buffer_size)
    return train_ds

@tf.function
def get_image(path, name, h, w, scaling=1):
    path = tf.stack([[path], tf.convert_to_tensor([name])], axis=0)
    file_path = tf.strings.reduce_join(path, separator="/")
    file = tf.io.read_file(file_path)
    img = decode_img(file, h, w)
    img = img * tf.cast(scaling, tf.float32)
    return img[...,0:3]

@tf.function
def get_gt(path, n):
    path = tf.stack([[path], tf.convert_to_tensor(['gt.txt'])], axis=0)
    file_path = tf.strings.reduce_join(path, separator="/")
    line = next(iter(tf.data.TextLineDataset(file_path)))
    mask = tf.io.decode_csv(line, list(np.ones(n*3)), field_delim=' ', use_quote_delim=False)

    return tf.convert_to_tensor(mask)

@tf.function
def decode_img(img, h, w):
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [h, w])

# @tf.function
def encode_mask(mask, n):
    ind = tf.cast(tf.round(mask / 255 * (n-1)), tf.uint8)
    return tf.one_hot(tf.squeeze(ind, axis=-1), n, axis=-1)
    indices = tf.linspace(0, 255, n)
    mask = tf.cast(mask, indices.dtype)
    e = tf.eye(n, dtype=indices.dtype)

    h,w = mask.shape[-3:-1]

    m = tf.reshape(mask[..., :1], (h*w, 1))
    def f(elem):
        i = tf.where(elem == indices)[0][0]
        return e[i]

    mask = tf.map_fn(f, m)
    mask = tf.reshape(mask, (h, w, -1))

    return mask

@tf.function
def process_path(path, scaling, type, reg, h, w, n, ret_gt, round, gamma, map_fn):
    with tf.device('/device:GPU:0'):
        img_name = "img.png" #if reg else 'img_corrected_1.png'
        if type == dp.TEST_INV:
            img_name = "img.png"
        if reg:
            mask_name = "gt.png"
        else:
            mask_name = 'gt_mask.png'

        mask = get_image(path, mask_name, h, w)
        # load the raw data from the file as a string
        img = get_image(path, img_name, h, w, scaling=scaling)

        if gamma:
            img = (tf.pow(img, 1.0 / 2.2) *
                           255.0)

        if type == dp.TRAIN:
            img, mask = augmentations.augment(img, mask)

        img = tf.image.resize(img, [h, w])
        mask = tf.image.resize(mask, [h, w])

        if ret_gt or reg:
            gt = get_gt(path, n)
            gt = tf.cast(gt, img.dtype)
        if reg:
            ret = (img, mask, gt)
        else:
            mask = tf.reduce_max(mask, axis=-1, keepdims=True)
            if round:
                mask = tf.quantization.fake_quant_with_min_max_args(mask * 255, 0, 255, 2, narrow_range=True)
                mask = encode_mask(mask, n)
                mask.set_shape((h, w, n))

            if ret_gt:
                ret = (img, mask, gt)
            else:
                ret = (img, mask, None)
        ret = map_fn(*ret)
    return ret

if __name__ == '__main__':
    disk = '/media/donik/Disk'
    paths = load_paths(disk + '/CubeN_bounding', 'list.txt')
    ds_base = dataset(paths, type=dp.TEST, bs=1, n=3, cache=False, regression=False, round=False, gt=True, shuffle_buffer_size=1)
    for data in iter(ds_base):
        img, mask, gt = data
        img = img
        mask = tf.quantization.fake_quant_with_min_max_args(mask * 255, 0, 255, 2, narrow_range=True)
        visualizer.visualize(mask[..., 0])
        mask_e = encode_mask(mask, 3)
        visualizer.visualize([mask[0, :, :, 0], mask_e[0]])

