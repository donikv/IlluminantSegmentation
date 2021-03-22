import tensorflow as tf
import tensorflow_addons as tfa
import random
import numpy as np

def augment(*images: tf.Tensor, mask_image = False):
    p1 = np.random.uniform((), 0, 1)
    p2 = np.random.uniform((), 0, 1)
    # p3 = tf.random.uniform((), 0, 1)
    random_state = np.random.RandomState(None)

    if images[0].shape.__len__() == 4:
        random_angles = tf.random.uniform(shape=(tf.shape(images[0])[0],), minval=-np
                                          .pi / 4, maxval=np.pi / 4)
    if images[0].shape.__len__() == 3:
        random_angles = tf.random.uniform(shape=(), minval=-np
                                          .pi / 4, maxval=np.pi / 4)


    crop = random.uniform(0.7, 0.9)
    random_rot = tf.random.uniform([], minval=0, maxval=3,dtype=tf.int32)
    if mask_image:
        size = images[0].shape[1:3] if images[0].shape.__len__() == 4 else images[0].shape[0:2]
        random_cutout_offset = tf.random.uniform(shape=(2,), minval=0, maxval=min(size), dtype=tf.int32)
        random_cutout_size = tf.random.uniform(shape=(2,), minval=0, maxval=min(size) // 2, dtype=tf.int32)
        # random_mask = create_mask(size)
    def aug(image):
        if p1 > 0.5:
            image = tf.image.flip_left_right(image)
        if p2 > 0.5:
            image = tf.image.flip_up_down(image)

        if mask_image:
            # image = tf.where(tf.expand_dims(random_mask, axis=-1) != 0, image, tf.zeros_like(image))
            image = tfa.image.cutout(image[tf.newaxis, :, :, :], random_cutout_size, random_cutout_offset)[0]

        image = tfa.image.rotate(image, random_angles)
        image = tf.image.rot90(image, random_rot)
        image = tf.image.central_crop(image, crop)

        # image = elastic_transform(image, image.shape[1] * 2, image.shape[1] * p2 / 5, image.shape[1] * p1 / 5, random_state)
        return image

    return list(map(aug, images))



def color(*images: tf.Tensor):
    rand_hue = tf.random.uniform((), 0, 0.3)
    rand_sat = tf.random.uniform((), 0.8, 2)
    rand_brihgt = tf.random.uniform((), 0, 0.2)
    rand_cont = tf.random.uniform((), 0.8, 1.5)

    def transform(img):
        img = tf.image.adjust_hue(img, rand_hue)
        img = tf.image.adjust_saturation(img, rand_sat)
        img = tf.image.adjust_brightness(img, rand_brihgt)
        img = tf.image.adjust_contrast(img, rand_cont)
        return img

    if len(images) == 1:
        return transform(images[0])

    images = list(map(transform, images))
    return images


import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return tf.constant(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape))


def create_mask(size=tf.constant((480, 480)), transpose=tf.constant(False), make_positive=True):
    if transpose:
        start = tf.random.uniform([], 0, size[0], tf.int32)
        line_index = tf.range(size[1], dtype=tf.int32)
        line = tf.zeros((size[1]), dtype=tf.int32)
        x_size = size[0]
        y_size = size[1]
    else:
        start = tf.random.uniform([], 0, size[1], tf.int32)
        line_index = tf.range(size[0], dtype=tf.int32)
        line = tf.zeros((size[0]), dtype=tf.int32)
        x_size = size[1]
        y_size = size[0]

    y_current = -1
    x_current = start

    prob = tf.random.uniform([5], 0, 100, tf.float32)

    def cond(x_size, x_current, y_size, y_current, line, line_index):
        return tf.math.less(y_current, y_size)

    def body(x_size, x_current, y_size, y_current, line, line_index):
        select = tf.constant([0, 1, 2, 3, 4])
        sample = tf.squeeze(tf.random.categorical(tf.math.log([prob]), 1))
        step = select[sample]
        if tf.equal(step, 0):
            x_current -= 1
            replace_value = tf.ones((y_size), dtype=tf.int32) * x_current
            line = tf.where(line_index == y_current, replace_value, line)
        elif tf.equal(step, 1):
            x_current -= 1
            y_current += 1
            replace_value = tf.ones((y_size), dtype=tf.int32) * x_current
            line = tf.where(line_index == y_current, replace_value, line)
        elif tf.equal(step, 2):
            y_current += 1
            replace_value = tf.ones((y_size), dtype=tf.int32) * x_current
            line = tf.where(line_index == y_current, replace_value, line)
        elif tf.equal(step, 3):
            x_current += 1
            y_current += 1
            replace_value = tf.ones((y_size), dtype=tf.int32) * x_current
            line = tf.where(line_index == y_current, replace_value, line)
        else:
            x_current += 1
            replace_value = tf.ones((y_size), dtype=tf.int32) * x_current
            line = tf.where(line_index == y_current, replace_value, line)

        return x_size, x_current, y_size, y_current, line, line_index

    _, _, _, _, line, _ = tf.while_loop(cond, body, [x_size, x_current, y_size, y_current, line, line_index],
                                        parallel_iterations=10)

    if transpose:
        x = tf.tile([tf.range(0, size[0], dtype=tf.int32)], [size[1], 1])
        rez = x - tf.expand_dims(line, 1)
        rez = tf.transpose(rez)
    else:
        x = tf.tile([tf.range(0, size[1], dtype=tf.int32)], [size[0], 1])
        rez = x - tf.expand_dims(line, 1)

    ones = tf.ones(size, tf.int32)
    zeros = tf.zeros(size, tf.int32)

    rez = tf.where(rez >= 0, ones, zeros)

    if make_positive:
        nz = tf.cast(tf.math.count_nonzero(rez), tf.int32)
        if nz < size[0] * size[1] // 2:
            rez = 1 - rez

    return rez

if __name__ == '__main__':
    import time
    from general.utils import visualizer

    start = time.time_ns()
    mask = create_mask()
    end = time.time_ns()
    dur1 = end - start
    visualizer.visualize([mask])

    start = time.time_ns()
    mask = create_mask(make_positive=False)
    end = time.time_ns()
    dur2 = end - start

    print(f"{dur1 / 1000 / 1000}, {dur2/1000/1000}")