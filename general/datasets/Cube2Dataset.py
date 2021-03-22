import tensorflow as tf
import os
import numpy as np
from general.processing import augmentations, data_processing as dp

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


def dataset(im_paths, type=dp.TRAIN, bs=2, cache=True, regression=False, gt=False, gt_mask=False, round=True, gamma=False, camera=None, scene_type=None, new_annotation=False, h=IMG_HEIGHT, w=IMG_WIDTH, map_fn=lambda *x: x, shuffle_buffer_size=100, resample=1):
    if resample > 1:
        im_paths_indoor = np.array(list(filter(lambda x: x.find('indoor') != -1, im_paths)))
        im_paths_nighttime = np.array(list(filter(lambda x: x.find('nighttime') != -1, im_paths)))
        for i in range(resample):
            im_paths = np.concatenate([im_paths, im_paths_indoor, im_paths_nighttime], axis=-1)
    if camera is not None:
        im_paths = np.array((list(filter(lambda x: x.find(camera) != -1, im_paths))))
    if scene_type is not None:
        im_paths = np.array((list(filter(lambda x: x.find(scene_type) != -1, im_paths))))
    list_ds = tf.data.Dataset.from_tensor_slices(im_paths)
    scaling = list_ds.map(lambda x: 1 if gamma else 4)
    list_ds = list_ds.zip((list_ds, scaling))
    labeled_ds = list_ds.map(lambda x,s: process_path(x, s, type, regression, h, w, gt, gt_mask, round, gamma, new_annotation, map_fn), num_parallel_calls=AUTOTUNE)
    train_ds = dp.prepare_for_training(labeled_ds, bs=bs, type=type, cache=cache, shuffle_buffer_size=shuffle_buffer_size)
    if resample > 1 or camera is not None or scene_type is not None:
        return train_ds, im_paths
    return train_ds

@tf.function
def get_image(path, name, h, w, new, scaling=1):
    if name == 'gt_mask.png' and new:
        return get_mask(path, name, h, w)[...,0:3]
    path = tf.stack([[path], tf.convert_to_tensor([name])], axis=0)
    file_path = tf.strings.reduce_join(path, separator="/")
    file = tf.io.read_file(file_path)
    img = decode_img(file, h, w)
    return img[...,0:3]


@tf.function
def get_mask(path, name, h, w, scaling=1):
    path = tf.stack([[path], tf.convert_to_tensor([name])], axis=0)
    file_path = tf.strings.reduce_join(path, separator="/")
    file = tf.io.read_file(file_path)
    img = decode_img(file, h, w)
    img = img * tf.cast(scaling, tf.float32)
    return img[...,0:3]


@tf.function
def get_gt(path, csv_fmt=[1., 1., 1., 1., 1., 1.]):
    path = tf.stack([[path], tf.convert_to_tensor(['gt.txt'])], axis=0)
    file_path = tf.strings.reduce_join(path, separator="/")
    line = next(iter(tf.data.TextLineDataset(file_path)))
    mask = tf.io.decode_csv(line, csv_fmt, field_delim=' ', use_quote_delim=False)

    return tf.convert_to_tensor(mask)

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
    m = tf.where(mask == tf.zeros_like(mask), tf.convert_to_tensor([0., 0., 1.]), m)
    mask = tf.where(mask == tf.ones_like(mask), tf.convert_to_tensor([1., 0., 0.]), m)
    mask = tf.where(tf.reduce_mean(img, axis=-1, keepdims=True) < tf.ones_like(mask) * 0.005, tf.convert_to_tensor([0., 1., 0.]), mask)
    return mask

@tf.function
def process_path(path, scaling, type, reg, h, w, ret_gt, ret_gt_mask, round, gamma, new, map_fn):
    with tf.device('/device:GPU:0'):
        img_name = "img.png" #if reg else 'img_corrected_1.png'
        if type == dp.TEST_INV:
            img_name = "img.png"
        if reg:
            mask_name = "gt.png"
        elif round:
            mask_name = 'gt_mask_round.png'
        else:
            mask_name = 'gt_mask.png'

        mask = get_image(path, mask_name, h, w, new)
        if reg:
            mask = mask / (tf.linalg.norm(mask, 2, -1, True) + 1e-7)
        if ret_gt_mask:
            gt_mask = get_image(path, "gt.png", h, w, new)
            gt_mask = gt_mask / (tf.linalg.norm(gt_mask, 2, -1, True) + 1e-7)
            mask = tf.concat([gt_mask, mask], axis=-1)
        # load the raw data from the file as a string
        img = get_image(path, img_name, h, w, new, scaling=scaling)

        if gamma:
            img = (tf.pow(img, 1.0 / 2.2) *
                   255.0)

        if type == dp.TRAIN:
            img, mask = augmentations.augment(img, mask)

        img = tf.image.resize(img, [h, w])
        mask = tf.image.resize(mask, [h, w])

        if ret_gt or reg:
            gt = get_gt(path, csv_fmt=[1., 1., 1.] if new else [1., 1., 1., 1., 1., 1.])
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

            if ret_gt:
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
    ds_base = dataset(paths, type=dp.TEST, bs=1, cache=False, regression=False, round=False, gt=True, new_annotation=True, shuffle_buffer_size=1)
    for data in iter(ds_base):
        img, mask, gt = data
        img = img
        mask = encode_mask(mask, img)

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