import tensorflow as tf
from general.processing import histogram, patcher
from general.utils import visualizer

AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN = 'train'
TEST = 'test'
TEST_INV = 'test2'
VALID = 'valid'

@tf.function
def patch_image(img, mask, p, img_shape, mask_shape):
    mask = patcher.extract_patches(tf.convert_to_tensor(mask), p, mask_shape)
    img = patcher.extract_patches(tf.convert_to_tensor(img), p, img_shape)

    return img, mask

@tf.function
def reduce_mask(mask, depth):
    mask = tf.reduce_mean(mask, axis=(1, 2))
    if depth is None:
        mask = tf.argmax(mask, axis=1)
        mask = tf.expand_dims(mask, axis=-1)
    else:
        mask, masky = histogram.bin(mask, depth)
        mask = mask[..., 0] * depth + mask[..., 1]
        mask = tf.expand_dims(mask, axis=-1)
    return mask

@tf.function
def bin_images(images: tf.Tensor, depth, shrink=1):
    def f(img):
        bin, Iy = histogram.bin(img, depth, shrink=shrink)
        Iy = tf.cast(Iy * 255, tf.int32)
        return tf.stack((bin[..., 0], bin[..., 1], Iy), axis=-1)

    images = tf.map_fn(f, images, name="img2bin", dtype=tf.int32)
    return images

@tf.function
def hist_images(images: tf.Tensor, depth, shrink=1):
    def f(img):
        return histogram.flatten_image(img, depth, shrink)
    images = tf.map_fn(f, images, name="img2hist")
    return images

@tf.function
def to_uv_images(images):
    def f(img):
        return histogram.to_uv(img)[0]
    images = tf.map_fn(f, images, name="img2uv")
    return images

@tf.function
def patch_dataset(dataset: tf.data.Dataset, patch_size, img_shape, mask_shape):
    dataset = dataset.map(
            lambda *x: (*patch_image(x[0], x[1], patch_size, img_shape, mask_shape), *x[2:]),
            num_parallel_calls=AUTOTUNE)
    return dataset

@tf.function
def timestamp_dataset(dataset: tf.data.Dataset, timestamps, img_shape, mask_shape):
    dataset = dataset.map(
            lambda *x: (tf.reshape(x[0], (-1, timestamps, *img_shape[1:])),
                        tf.reshape(x[1], (-1, timestamps, *mask_shape[1:])),
                        *x[2:]),
            num_parallel_calls=AUTOTUNE)
    return dataset

@tf.function
def reduce_dataset(dataset: tf.data.Dataset, depth):
    dataset = dataset.map(
        lambda *x: (x[0], reduce_mask(x[1], depth), *x[2:]), num_parallel_calls=AUTOTUNE
    )
    return dataset

@tf.function
def histogram_dataset(dataset: tf.data.Dataset, depth, shrink=1):
    dataset = dataset.map(
        lambda *x: (hist_images(x[0], depth, shrink), *x[1:]), num_parallel_calls=AUTOTUNE
    )
    return dataset


def binned_dataset(dataset: tf.data.Dataset, depth, shrink=1):
    dataset = dataset.map(
        lambda *x: (bin_images(x[0], depth, shrink), *x[1:]), num_parallel_calls=AUTOTUNE
    )
    return dataset


def duplicate_visual(dataset: tf.data.Dataset):
    dataset = dataset.map(
        lambda *x: (*x, x[0]), num_parallel_calls=AUTOTUNE
    )
    return dataset


def uv_dataset(dataset: tf.data.Dataset):
    dataset = dataset.map(
        lambda *x: (to_uv_images(x[0]), *x[1:]), num_parallel_calls=AUTOTUNE
    )
    return dataset


#CCC
@tf.function
def gaussian_kernel(size: int,
                    mean: float = 0.,
                    std: float = 1.,
                   ):
    """Makes 2D gaussian Kernel for convolution."""

    d = tf.compat.v1.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

    gauss_kernel = tf.einsum('i,j->ij',
                                  vals,
                                  vals)

    return gauss_kernel / (tf.reduce_sum(gauss_kernel) + 1e-7)

@tf.function
def blur_kernel(size: int):
    filter = tf.ones((3,3), dtype=tf.float32)
    filter = filter / (size**2)
    return filter

# @tf.function
def filter_img(image, kernel, channels=3):
    if len(image.shape) == 3:
        image = tf.expand_dims(image, axis=0)

    kernel = kernel[:, :, tf.newaxis, tf.newaxis]
    if channels != 1:
        kernel = tf.tile(kernel, [1, 1, channels, 1])  # ks*ks*3*1
        # Pointwise filter that does nothing
    pointwise_filter = tf.eye(channels, batch_shape=[1, 1])
    return tf.nn.separable_conv2d(image, kernel, pointwise_filter,
                                strides=[1, 1, 1, 1], padding='SAME')
    # else:
    #     return tf.nn.conv2d(image, kernel, strides=[1,1,1,1], padding='SAME')

import numpy as np
@tf.function
def __process_images__(Is, indecies=[0,1,2,3,4]):
    def f(I):
        # print(I.shape)
        I4 = filter_img(I*I, blur_kernel(3)) - filter_img(I, blur_kernel(3))**2
        I4 = tf.maximum(I4, 0.)
        I4 = tf.sqrt(I4)
        I4 = tf.squeeze(I4)
        I3 = tf.maximum(filter_img(I**4, blur_kernel(11)), 0.)
        I3 = tf.sqrt(tf.sqrt(I3))
        I3 = tf.squeeze(I3)
        kernel = tf.constant([[0,-1,0],
                              [-1,5,-1],
                              [0,-1,0]], dtype=tf.float32)
        I2 = filter_img(I, kernel)
        I2 = tf.squeeze(tf.maximum(0., I2))

        dif_kernel = tf.constant([[-1,-1,-1],
                              [-1,8,-1],
                              [-1,-1,-1]], dtype=tf.float32) / 8
        I5 = filter_img(I, dif_kernel)
        I5 = tf.squeeze(tf.abs(I5))
        I = tf.squeeze(I)

        images = np.array([I, I2, I3, I4, I5])
        images = list(images[indecies])

        images = tf.stack(images)
        tf.debugging.assert_all_finite(
            images, 'Images must not contain nan values', name=None
        )

        return images
    return f(Is)

@tf.function
def __hr__(Is, d, s):
    def f(imgs):
        def l(img):
            h = histogram.flatten_image(img, d, s, remove_black=True)
            return tf.reshape(h, (d, d, 1))
        return tf.map_fn(l, imgs)
    return f(Is)


def multi_repr_dataset(dataset: tf.data.Dataset, depth, shrink=1):
    def f(*x):
        images = __process_images__(x[0])
        hists = __hr__(images, depth, shrink)
        return (hists, *x[1:], images)
    dataset = dataset.map(
        f, num_parallel_calls=AUTOTUNE
    )
    return dataset


from general.training.losses import cosine_similarity


@tf.function
def multi_representation_histogram(*x, depth, shrink, known_illuminant=False):
    images = __process_images__(x[0])
    hists = __hr__(images, depth, shrink)

    if known_illuminant:
        ct = histogram.color_table(depth)
        gt1 = visualizer.create_mask(x[-1][:3], (depth, depth))
        ct = tf.sqrt(cosine_similarity(ct, gt1, keepdims=True))
        ct = tf.tile(ct[tf.newaxis, :, :], (hists.shape[0], 1, 1, 1))
        print(ct.shape, hists.shape)
        hists = hists * ct

    return (hists, *x[1:], images)

@tf.function
def transformation_histogram(*x, depth, shrink, known_illuminant=False):
    images = __process_images__(x[0], indecies=[0,3])
    hists = __hr__(images, depth, shrink)
    images = None
    images_corrected = __process_images__(x[-1])
    hists2 = __hr__(images_corrected, depth, shrink)

    if known_illuminant:
        ct = histogram.color_table(depth)
        gt1 = visualizer.create_mask(x[-1][:3], (depth, depth))
        ct = tf.sqrt(cosine_similarity(ct, gt1, keepdims=True))
        ct = tf.tile(ct[tf.newaxis, :, :], (hists.shape[0], 1, 1, 1))
        print(ct.shape, hists.shape)
        hists = hists * ct

    return (hists, hists2, *x[1:], x[0])

@tf.function
def image_histogram_mapping_segmentation(*x, uv=False, normalize=False, reg_head=True, weighted=False, multioutput=False, clustering=False, correct=True):
    mask = x[1]
    img = x[0]
    gt = tf.cast(x[-1], dtype=tf.float32)
    if gt != None and correct:
        gt1 = tf.reshape(gt, (-1, 3))
        img = img / gt1[0]
        gt = gt1[1] / gt1[0]
        gt = tf.reshape(gt, (3,))
    h = tf.image.rgb_to_hsv(img)[..., 0]
    img_uv, Iy = histogram.to_uv(img)

    Iy = tf.where(Iy < 0.01, 0., Iy)
    if normalize:
        Iy = tf.image.per_image_standardization(Iy[:, :, tf.newaxis])
        Iy = Iy[..., 0]

    img_hy = tf.stack([Iy, h, Iy], axis=-1)
    img_uv = tf.stack([Iy, img_uv[..., 0], img_uv[..., 1]], axis=-1)
    img = img_uv if uv else img

    if multioutput:
        features = tf.concat((img, img_hy), axis=-1)

        if weighted:
            return features, mask, gt, tf.where(Iy == 0., 0., 1.), tf.ones((1,))
        return features, mask, gt

    elif clustering:
        gt1, gt2 = gt[:3], gt[3:]
        gt1 = gt1 / (tf.linalg.norm(gt1, axis=-1, ord=2, keepdims=True) + 1e-7)
        gt2 = gt2 / (tf.linalg.norm(gt2, axis=-1, ord=2, keepdims=True) + 1e-7)
        gt1 = visualizer.create_mask(gt1, sz =img.shape[-3:-1])
        gt2 = visualizer.create_mask(gt2, sz =img.shape[-3:-1])
        features = tf.concat((img, img_hy, gt1, gt2), axis=-1)
        # gt_mask = mask[..., :3]
        # gt_mask = gt_mask / (tf.linalg.norm(gt_mask, axis=-1, ord=2, keepdims=True) + 1e-7)

        # mask = mask[..., -1:]

        if weighted:
            return features, mask, tf.where(Iy == 0., 0., 1.)
        return features, mask

    elif reg_head:
        features = tf.concat((img, img_hy), axis=-1)
        gt_uv, gty = histogram.to_uv(tf.reshape(gt, (-1, 3)))
        gt_uv = tf.reshape(gt_uv, (2,))
        gt = gt_uv if uv else gt
        gt = [mask, gt] if multioutput else tf.concat([mask, tf.broadcast_to(gt, (*mask.shape[:-1], *gt.shape))], axis=-1)
    else:
        features = tf.concat((img, img_hy), axis=-1)
        gt = x[1]
    if weighted:
        return features, gt, tf.where(Iy == 0., 0., 1.)
    return features, gt


@tf.function
def ese_mapping(*x, corrected_input=False, pivot_gt=None, repeat_gt2=False, invert_mask=False, rb=False):
    mask = x[1]
    img = x[0]
    gt = tf.cast(x[-1], dtype=tf.float32)
    gt = tf.reshape(gt, (-1,3))

    gt = tf.nn.l2_normalize(gt, axis=-1)

    h = tf.image.rgb_to_hsv(img)[..., 0]
    if rb:
        _, Iy = histogram.to_rb(img)
        # img = tf.stack([img[..., 0], img[..., 1], Iy],axis=-1)
        gt, _ = histogram.to_rb(gt)
    else:
        _, Iy = histogram.to_uv(img)

    Iy = tf.where(Iy < 0.01, 0., Iy)
    img_hy = tf.stack([Iy, h, Iy], axis=-1)

    if pivot_gt is not None:
        pivot_gt = tf.constant(pivot_gt)
        c1 = cosine_similarity(gt[0], pivot_gt)
        c2 = cosine_similarity(gt[1], pivot_gt)
        if c2 < c1:
            gt = tf.stack([gt[1], gt[0]], axis=0)
            mask = tf.ones_like(mask) - mask


    masks = [mask] if not invert_mask else [mask,  tf.ones_like(mask) - mask]
    masks_weights = [tf.where(Iy == 0., 0., 1.)] if not invert_mask else [tf.where(Iy == 0., 0., 1.), tf.where(Iy == 0., 0., 1.)]
    gts = [gt[0]] if not repeat_gt2 else [gt[0], gt[1]]
    gt_weights = [tf.ones((1,))] if not repeat_gt2 else [tf.ones((1,)), tf.ones((1,))]

    if corrected_input:
        img_cor = img / gt[0]
        features = tf.concat((img, img_cor, img_hy), axis=-1)
        return features, \
               mask, gt[0], gt[1], \
               tf.where(Iy == 0., 0., 1.), tf.ones((1,)), tf.ones((1,))
    else:
        features = tf.concat((img, img_hy), axis=-1)
        return (features, \
                *gts, *masks, gt[0], gt[1], \
                *gt_weights, *masks_weights, tf.ones((1,)), tf.ones((1,)))


@tf.function
def multi_segmentation_mapping(*x, rb=False, clustering=False, regression=False):
    mask = x[1]
    img = x[0]
    gt = x[-1]
    gt = tf.reshape(gt, (-1,3))
    gt = tf.nn.l2_normalize(gt, axis=-1)
    gt = tf.reshape(gt, (-1,))

    h = tf.image.rgb_to_hsv(img)[..., 0]
    if rb:
        _, Iy = histogram.to_rb(img)
        # img = tf.stack([img[..., 0], img[..., 1], Iy],axis=-1)
        gt, _ = histogram.to_rb(gt)
    else:
        _, Iy = histogram.to_uv(img)

    Iy = tf.where(Iy < 0.01, 0., Iy)
    img_hy = tf.stack([Iy, h, Iy], axis=-1)

    if clustering:
        features = tf.concat((img, img_hy), axis=-1)
        gt = visualizer.create_mask(gt, sz =img.shape[-3:-1])
        mask = tf.concat([tf.cast(mask, float), gt], axis=-1)
    elif regression:
        features = tf.concat((img, img_hy), axis=-1)
        mask = mask / (tf.linalg.norm(mask, axis=-1, ord=2, keepdims=True) + 1e-7)
    else:
        img_cor = img / gt[0]
        features = tf.concat((img_cor, img_hy), axis=-1)

    return features, \
           mask, \
           tf.where(Iy == 0., 0., 1.)


@tf.function
def multi_ese_mapping(*x, corrected_input=False, repeat_gt2=False, invert_mask=False):
    mask = x[1]
    img = x[0]
    gt = x[-1]
    gt = tf.reshape(gt, (-1,3))

    gt = tf.nn.l2_normalize(gt, axis=-1)

    h = tf.image.rgb_to_hsv(img)[..., 0]
    _, Iy = histogram.to_uv(img)

    Iy = tf.where(Iy < 0.01, 0., Iy)
    img_hy = tf.stack([Iy, h, Iy], axis=-1)

    masks = mask
    masks_weights = tf.where(Iy == 0., 0., 1.)
    gts = tf.reshape(gt, (-1,))
    gt_weights = tf.ones((1,))

    if corrected_input:
        img_cor = img / gt[0]
        features = tf.concat((img, img_cor, img_hy), axis=-1)
        return features, \
               mask, gts, \
               tf.where(Iy == 0., 0., 1.), tf.ones((1,))
    else:
        features = tf.concat((img, img_hy), axis=-1)
        return (features, \
                gt[0], masks, gts, \
                gt_weights, masks_weights, gt_weights)


def multioutput_labels_mapping(dataset: tf.data.Dataset, count=2):
    return dataset.map(lambda *x: (x[0], x[1:1+count], x[1+count:]))

@tf.function
def image_histogram_mapping(*x, add_histogram=True):
    img_rb, Iy = histogram.to_rb(x[0])
    gt = tf.convert_to_tensor(x[-1])
    if len(gt.shape) > 1:
        gt = gt[0]

    h = tf.image.rgb_to_hsv(x[0])[..., 0]
    Iy = tf.where(Iy < 0.01, 0., Iy)
    img_hy = tf.stack([Iy, h, Iy], axis=-1)


    features = tf.concat((x[0], img_hy), axis=-1)
    return features if add_histogram else x[0], gt

@tf.function
def estimation_mapping(*x):
    gt = tf.convert_to_tensor(x[-1])
    gt = tf.nn.l2_normalize(gt, axis=-1)
    return x[0], gt


def prepare_for_training(ds, type, cache=True, shuffle_buffer_size=100, bs=2):

    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    if type == TRAIN:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    if type != TEST and type != TEST_INV:
        # Repeat forever
        ds = ds.repeat()

    if bs > 0:
        ds = ds.batch(bs)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def encode_mask_np(mask, img):
    m = np.where(mask != np.ones_like(mask), np.array([0., 1., 0.]), mask)  # BLACK CLASS
    m = np.where(mask == np.zeros_like(mask), np.array([0., 0., 1.]), m)
    mask = np.where(mask == np.ones_like(mask), np.array([1., 0., 0.]), m)
    mask = np.where(img == np.zeros_like(mask), np.array([0., 1., 0.]), mask)
    return mask

if __name__ == '__main__':
    from general.datasets import Cube2Dataset

    image = Cube2Dataset.get_image("/media/donik/Disk/Cube2/outdoor/nikon_d7000/outdoor1/1", 'img.png', 256, 512)
    gt = Cube2Dataset.get_image("/media/donik/Disk/Cube2/outdoor/nikon_d7000/outdoor1/1", 'gt.png', 256, 512)
    gtm = Cube2Dataset.get_image("/media/donik/Disk/Cube2/outdoor/nikon_d7000/outdoor1/1", 'gt_mask.png', 256, 512)
    gts = Cube2Dataset.get_gt("/media/donik/Disk/Cube2/outdoor/nikon_d7000/outdoor1/1")

    packed = image_histogram_mapping_segmentation(image, tf.concat((gt, gtm), axis=-1)[...,:4], gts, uv = False,normalize = False,reg_head = False,multioutput = False,weighted = True,clustering=True, correct=False)

