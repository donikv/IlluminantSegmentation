import tensorflow as tf
import numpy as np
c = 3
h = 1024
p = 128

@tf.function
def extract_patches(image, p, shape=None):
    if shape:
        bs, h, w, c = shape
    else:
        bs, h, w, c = image.shape

    # Image to Patches Conversion
    pad = [[0,0],[0,0]]
    patches = tf.space_to_batch_nd(image, [p, p], pad)
    patches = tf.split(patches, p * p, 0)
    patches = tf.stack(patches, 3)
    patches = tf.reshape(patches, [bs * (h//p) * (w//p), p, p, c])

    return patches

@tf.function
def combine_pathces(patches, p, shape):
    h, w, c = shape
    pad = [[0, 0], [0, 0]]
    patches_proc = tf.reshape(patches, [-1, h//p, w//p, p * p, c])
    patches_proc = tf.split(patches_proc, p*p, 3)
    patches_proc = tf.stack(patches_proc, axis=0)
    patches_proc = tf.reshape(patches_proc, [-1, h//p, w//p, c])

    reconstructed = tf.batch_to_space(patches_proc, [p, p], pad)
    return reconstructed

def expand(patches, p):
    b = tf.constant([1, p*p])
    return tf.tile(patches, b)