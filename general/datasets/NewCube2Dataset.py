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
    paths = np.array(list(map(lambda x: os.path.join(root, x[2:] if x.startswith('./') else x), paths)))
    return paths


def timestamped_histogram_dataset(dataset, p, depth, bs):
    mask_shape = (2, IMG_HEIGHT // 2, IMG_WIDTH // 2, 3)
    timestamps = IMG_HEIGHT // 2 * IMG_WIDTH // 2 // p ** 2
    ds = dp.reduce_dataset(dp.histogram_dataset(dp.patch_dataset(dataset, p, mask_shape, mask_shape), depth), None)
    ds = dp.timestamp_dataset(ds, timestamps, (timestamps * bs, depth**2), (timestamps * bs, 1))
    return ds


def dataset(im_paths, type=dp.TRAIN, bs=2, cache=True, regression=False, gt=False, gt_mask=False, round=True, gamma=False, camera=None, scene_type=None, h=IMG_HEIGHT, w=IMG_WIDTH, map_fn=lambda *x: x, shuffle_buffer_size=100, resample=1):
    im_paths = np.array(list(filter(lambda x: x.find('outdoor2/0') == -1, im_paths)))
    if resample > 1:
        im_paths_indoor = np.array(list(filter(lambda x: x.find('indoor') != -1, im_paths)))
        im_paths_nighttime = np.array(list(filter(lambda x: x.find('nighttime') != -1, im_paths)))
        for i in range(resample):
            im_paths = np.concatenate([im_paths, im_paths_indoor, im_paths_nighttime], axis=-1)
    if camera is not None:
        im_paths = np.array((list(filter(lambda x: x.find(camera) != -1, im_paths))))
    if scene_type is not None:
        im_paths = np.array((list(filter(lambda x: x.find(scene_type) != -1, im_paths))))
    if gt or regression:
        gts = [np.flip(np.loadtxt(x + "/gt.txt"), axis=0).reshape((-1,)) for x in im_paths]
    else:
        gts = [None for x in im_paths]
    gts_ds = tf.data.Dataset.from_tensor_slices(np.array(gts))
    list_ds = tf.data.Dataset.from_tensor_slices(im_paths)
    scaling = list_ds.map(lambda x: tf.constant(1.0, dtype=tf.float32) if gamma else tf.constant(4.0, dtype=tf.float32))
    list_ds = list_ds.zip((list_ds, scaling, gts_ds))
    labeled_ds = list_ds.map(lambda x,s,gt: process_path(x, s, type, regression, h, w, gt, gt_mask, round, gamma, map_fn), num_parallel_calls=AUTOTUNE)
    train_ds = dp.prepare_for_training(labeled_ds, bs=bs, type=type, cache=cache, shuffle_buffer_size=shuffle_buffer_size)
    if resample > 1 or camera is not None or scene_type is not None:
        return train_ds, im_paths
    return train_ds

@tf.function
def get_image(path, name, h, w, scaling=tf.constant(1.0, dtype=tf.float32)):
    path = tf.stack([[path], tf.convert_to_tensor([name])], axis=0)
    file_path = tf.strings.reduce_join(path, separator="/")
    file = tf.io.read_file(file_path)
    img = decode_img(file, h, w)
    img_norm = tf.linalg.norm(img) / 10
    img = img * scaling / img_norm
    img = tf.where(tf.reduce_max(img, axis=-1, keepdims=True) >= 0.99, tf.zeros_like(img), img)
    return img[...,0:3]


@tf.function
def get_mask(path, name, h, w):
    path = tf.stack([[path], tf.convert_to_tensor([name])], axis=0)
    file_path = tf.strings.reduce_join(path, separator="/")
    file = tf.io.read_file(file_path)
    img = tf.image.decode_png(file, channels=3, dtype=tf.uint8)
    img = tf.reduce_max(img) - img
    return img[...,0:3]


@tf.function
def get_gt(path, gts):
    gt = gts[path]
    gt = tf.convert_to_tensor(gt, dtype=tf.float32)
    print(gt.dtype)
    return tf.cast(gt, dtype=tf.float32)

@tf.function
def decode_img(img, h, w):
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [h, w])

@tf.function
def decode_mask(mask, h, w):
    mask = tf.image.decode_png(mask, channels=3) * 255
    mask = tf.where(mask != 255, 0, 255)
    return tf.image.resize(mask, [h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# @tf.function
def encode_mask(mask, img):
    m = tf.where(mask != tf.ones_like(mask), tf.convert_to_tensor([0., 1., 0.]), mask)# BLACK CLASS
    m = tf.where(tf.abs(mask - tf.zeros_like(mask)) < 1e-1, tf.convert_to_tensor([0., 0., 1.]), m)
    mask = tf.where(tf.abs(mask - tf.ones_like(mask)) < 1e-1, tf.convert_to_tensor([1., 0., 0.]), m)
    mask = tf.where(tf.linalg.norm(img, axis=-1, keepdims=True) < tf.ones_like(mask) * 1e-7, tf.convert_to_tensor([0., 1., 0.]), mask)
    return mask

@tf.function
def process_path(path, scaling, type, reg, h, w, gt, ret_gt_mask, round, gamma, map_fn):
    with tf.device('/device:GPU:0'):
        img_name = "img.png" #if reg else 'img_corrected_1.png'
        if type == dp.TEST_INV:
            img_name = "img.png"
        if reg:
            mask_name = "gt.png"
            mask = get_image(path, mask_name, h, w)
        elif round:
            mask_name = 'gt_mask_round.png'
            mask = get_image(path, mask_name, h, w)
        else:
            mask_name = 'gt_mask.png'
            mask = get_mask(path, mask_name, h, w)

        if reg:
            mask = mask / (tf.linalg.norm(mask, 2, -1, True) + 1e-7)
        if ret_gt_mask:
            gt_mask = get_image(path, "gt.png", h, w)
            gt_mask = gt_mask / (tf.linalg.norm(gt_mask, 2, -1, True) + 1e-7)
            mask = tf.concat([gt_mask, mask], axis=-1)
        # load the raw data from the file as a string
        img = get_image(path, img_name, h, w, scaling=scaling)

        if gamma:
            img = (tf.pow(img, 1.0 / 2.2) *
                   255.0)

        if type == dp.TRAIN:
            img, mask = augmentations.augment(img, mask)

        img = tf.image.resize(img, [h, w])
        mask = tf.image.resize(mask, [h, w])

        # if gt is not None:
        #     gt = get_gt(path, gts)
        if reg:
            ret = (img, mask, gt)
        else:
            if round:
                mask = encode_mask(mask, img)
                mask.set_shape((h, w, 3))
            else:
                if not ret_gt_mask:
                    mask = tf.reduce_max(mask, axis=-1, keepdims=True)
                else:
                    mask = mask[..., :4]

            if gt is not None:
                ret = (img, mask, gt)
            else:
                ret = (img, mask, None)
        ret = map_fn(*ret)
    return ret

# if __name__ == '__main__':
#
#     import visualizer
#     from functools import partial
#
#     paths = load_paths('/media/donik/Slowpoke/fax/CubeRelighted/', 'list_relighted_limited_gamut.txt')
#     # map_fn = partial(dp.transformation_histogram, depth=256, shrink=30)
#     ds = dataset(paths, gt=True, corrected=False, shuffle_buffer_size=1, type=dp.TEST)
#     # uv_ds = dp.uv_dataset(ds)
#     img_shape = (2, IMG_HEIGHT // 2, IMG_WIDTH // 2, 2)
#     mask_shape = (2, IMG_HEIGHT // 2, IMG_WIDTH // 2, 3)
#     # dsseg = dp.segmenation_features_dataset(ds)
#     for data in iter(ds):
#         img, mask, gt = data
#         visualizer.visualize(img)
#         gt = tf.reshape(gt, (-1, 2,3))
#         img2 = img[0] / gt[0, 1]
#         img1 = img[0] / gt[0, 0]
#         visualizer.visualize([img1, img2])
#         visualizer.visualize(mask)

if __name__ == '__main__':
    disk = '/media/donik/Disk'
    paths = load_paths(disk + '/Cube2_new', 'list_outdoor.txt')
    paths = np.array(list(filter(lambda x: x.find('outdoor6') != -1, paths)))
    ds_base = dataset(paths, type=dp.TEST, bs=1, cache=False, regression=False, round=False, gt=True, shuffle_buffer_size=1)
    for data in iter(ds_base):
        img, mask, gt = data
        norm = tf.linalg.norm(img, axis=(1,2,3))
        img = img * 16
        emask = encode_mask(mask, img)
        visualizer.visualize([img[0], mask[0, :, :, 0], emask[0], visualizer.create_mask(gt[0, :3], (10, 10)), visualizer.create_mask(gt[0, 3:], (10, 10))])

# p = 32
# ds2 = dp.patch_dataset(uv_ds, p, img_shape, mask_shape)
# ds3 = dp.reduce_dataset(ds2, None)
# ds4 = dp.reduce_dataset(dp.histogram_dataset(ds, 64), None)
# ds5 = dp.reduce_dataset(dp.histogram_dataset(dp.patch_dataset(ds, p, mask_shape, mask_shape), 64), None)
# ds6 = dp.timestamp_dataset(ds5, IMG_HEIGHT//2 * IMG_WIDTH//2 // p**2, (1024, 4096), (1024, 1))
# # data5 = next(iter(ds5))
# data6 = next(iter(ds6))
# data1 = next(iter(ds))
# data2 = next(iter(ds2))
# data3 = next(iter(ds3))
# data4 = next(iter(ds4))
#
#
# mask2 = patcher.combine_pathces(data2[1], p, mask_shape[1:])
#
# visualizer.visualize(data1[0])
# visualizer.visualize(data1[1])
# visualizer.visualize(mask2)
# mask3 = tf.squeeze(tf.one_hot(data3[1], 3))
# mask3 = patcher.expand(mask3, p)
# mask3 = patcher.combine_pathces(mask3, p, mask_shape[1:])
# visualizer.visualize(mask3)
# ct = histogram.color_table(depth=64)
# visualizer.visualize([tf.reshape(data4[0][0], (64,64, 1)) * ct])