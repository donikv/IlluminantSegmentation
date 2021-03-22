import tensorflow as tf
import numpy as np

EPS = 0.025
@tf.function
def to_uv(img):
    img = tf.clip_by_value(img, 0.01, 1)
    Iu = tf.math.log(img[...,1] / img[...,0])
    Iv = tf.math.log(img[...,1] / img[...,2])
    Iy = tf.math.sqrt(img[...,0]**2 + img[...,1]**2 + img[...,2]**2)
    img_uv = tf.stack([Iu, Iv], axis=-1)
    # img_uv = tf.where(tf.math.is_nan(img_uv), tf.zeros_like(img_uv), img_uv)
    return img_uv, Iy

@tf.function
def to_rb(img):
    inpt_rg = tf.math.divide_no_nan(img[..., 0], img[..., 1])
    inpt_bg = tf.math.divide_no_nan(img[..., 2], img[..., 1])
    img_rb = tf.stack((inpt_rg, inpt_bg), axis=-1)
    Iy = tf.math.sqrt(img[...,0]**2 + img[...,1]**2 + img[...,2]**2)
    return img_rb, Iy

@tf.function
def from_rb(img):
    img_rgb = tf.stack([img[..., 0], tf.ones_like(img[..., 0]), img[..., 1]], axis=-1)
    return img_rgb

def to_uv_np(img):
    img = np.clip(img, 0.01, 1)
    Iu = np.log(img[...,1] / img[...,0])
    Iv = np.log(img[...,1] / img[...,2])
    Iy = np.sqrt(img[...,0]**2 + img[...,1]**2 + img[...,2]**2)
    img_uv = np.stack([Iu, Iv], axis=-1)
    # img_uv = tf.where(tf.math.is_nan(img_uv), tf.zeros_like(img_uv), img_uv)
    return img_uv, Iy

@tf.function
def from_uv(img_uv, Iy=None):
    elu = tf.math.exp(-img_uv[...,0])
    elv = tf.math.exp(-img_uv[...,1])
    z = Iy if Iy is not None else tf.math.sqrt(elu**2 + elv**2 + 1)

    Ir = elu / z
    Ig = 1 / z
    Ib = elv / z
    img = tf.stack([Ir, Ig, Ib], axis=-1)
    return img


@tf.function
def bin(img, depth=16, center_u=0., center_v=0., eps=EPS, shrink=1, cs_func=to_uv, flat=False):
    if shrink != -1 and shrink != 1:
        img = tf.image.resize(img, (img.shape[0] // shrink, img.shape[1] // shrink))
    img_uv, Iy = cs_func(img)
    bn_cnt = depth
    # vr = tf.convert_to_tensor(list(range(-bn_cnt // 2, bn_cnt // 2)), dtype=tf.float32) * 0.05
    vr_u = [(-bn_cnt // 2 * eps) + center_u, (bn_cnt // 2 * eps) + center_u]
    vr_v = [(-bn_cnt // 2 * eps) + center_v, (bn_cnt // 2 * eps) + center_v]
    # vr = tf.scan(lambda x, y: x + y, vr) - 0.4
    bins_u = tf.histogram_fixed_width_bins(img_uv[..., 0], value_range=vr_u, nbins=bn_cnt)
    bins_v = tf.histogram_fixed_width_bins(img_uv[..., 1], value_range=vr_v, nbins=bn_cnt)
    bins = tf.stack([bins_u, bins_v], axis=-1)
    if flat:
        bins = bins[..., 0] * depth + bins[..., 1]

    return bins, Iy

@tf.function
def bn(x, depth, stack=True):
    ln = len(x.shape)
    if ln == 1:
        x = x[tf.newaxis, :]
    bins = tf.floor(x / EPS) + depth // 2
    bins = tf.clip_by_value(bins, 0, depth)
    if stack:
        bins = bins[..., 0] * depth + bins[..., 1]
    if ln == 1:
        bins = tf.reshape(bins, bins.shape[1:])
        print(bins.shape)
    return bins

@tf.function
def hist(bins, Iy, depth, remove_black=True):
    # bins = tf.image.resize(bins, (bins.shape[0] // 5, bins.shape[1] // 5))
    # Iy = tf.image.resize(tf.expand_dims(Iy, axis=-1), (Iy.shape[0] // 5, Iy.shape[1] // 5))
    bins_flat = tf.reshape(bins, (-1, 2))
    bins_flat = tf.cast(bins_flat, tf.int32)

    Iy_flat = tf.reshape(Iy, [-1])
    if remove_black:
        Iy_flat = tf.where(Iy_flat < 0.02, 0., Iy_flat)
        Iy_flat = Iy_flat / (tf.reduce_max(Iy_flat) + 1e-10)
    # bins_flat = tf.reshape(bins, [-1])
    bins_flat = tf.map_fn(lambda x: x[0] * depth + x[1], bins_flat)
    hist = tf.math.bincount(bins_flat, weights=Iy_flat, minlength=depth*depth)

    return hist

@tf.function
def encode_bins(bins, depth):
    # sz = bins.shape
    # bns_flat = tf.reshape(bins, (-1, 2))
    # bns_flat = tf.map_fn(lambda x: x[0] * depth + x[1], bns_flat)
    # bins = tf.reshape(bns_flat, (sz[0], sz[1], 1))
    bins = bins[...,0] * depth + bins[...,1]
    one_hot = tf.one_hot(bins, depth*depth, axis=-1)

    return one_hot

@tf.function
def decode_bins(one_hot, depth, center_u=0., center_v=0., eps=EPS, is_one_hot=True):
    if is_one_hot:
        idxs = tf.argmax(one_hot, axis=-1)
    else:
        idxs = tf.cast(one_hot, dtype=tf.int64)
    bin_u = tf.cast((idxs // depth) - depth // 2, dtype=tf.float32) * eps + center_u
    bin_v = tf.cast((idxs % depth) - depth // 2, dtype=tf.float32) * eps + center_v
    bins = tf.stack([bin_u, bin_v], axis=-1)

    return bins

@tf.function
def flatten_image(img, depth, shrink=1, remove_black=True):
    bins, Iy = bin(img, depth=depth, shrink=shrink)
    hist_uv = hist(bins, Iy, depth, remove_black)
    hist_uv = tf.sqrt(hist_uv/(tf.norm(hist_uv, 1) + 1e-7))

    return hist_uv

@tf.function
def color_table(depth, center_u=0., center_v=0., eps=EPS):
    bn_cnt = depth
    u = tf.range(-bn_cnt // 2 * eps, bn_cnt // 2 * eps, delta=eps) + center_u
    v = tf.range(-bn_cnt // 2 * eps, bn_cnt // 2 * eps, delta=eps) + center_v

    U, V = tf.meshgrid(u,v)
    grid = tf.stack((V,U), axis=-1)
    grid = from_uv(grid)
    return grid

# import visualizer
# # visualizer.visualize([color_table(7, center_u=0.3, center_v=0.7)])
# # visualizer.visualize([color_table(7, eps = 0.35)])
# # visualizer.visualize([color_table(7, eps=0.25)])
# img = color_table(7)
# h = flatten_image(img, 7)
# h = tf.reshape(h, (7, 7))
# hm = visualizer.create_mask(tf.expand_dims(h, 2))
# visualizer.visualize([img, h, hm])