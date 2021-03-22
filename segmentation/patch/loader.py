import pathos.multiprocessing as pp
from multiprocessing import cpu_count
import multiprocess as mp

import os
import numpy as np
from skimage.transform import *
from skimage.io import imread
from skimage.measure import block_reduce
from skimage.util import img_as_float


def to_uv_np(img):
    img = np.clip(img, 0.01, 1)
    Iu = np.log(img[...,1] / img[...,0])
    Iv = np.log(img[...,1] / img[...,2])
    Iy = np.sqrt(img[...,0]**2 + img[...,1]**2 + img[...,2]**2)
    img_uv = np.stack([Iu, Iv], axis=-1)
    # img_uv = tf.where(tf.math.is_nan(img_uv), tf.zeros_like(img_uv), img_uv)
    return img_uv, Iy

def cosine_similarity(l1, l2, eps=1e-7):
    l1_norm = np.linalg.norm(l1, axis=-1, keepdims=True) + eps
    l2_norm = np.linalg.norm(l2, axis=-1, keepdims=True) + eps
    l1l2 = np.sum(l1*l2, axis=-1, keepdims=True)
    cos = (l1l2) / (l1_norm*l2_norm)
    cos = np.clip(cos, -1, 1)
    return np.squeeze(np.arccos(cos))

def load_paths(root, list_name):
    paths = np.loadtxt(os.path.join(root, list_name), dtype=str)
    paths = np.array(list(map(lambda x: os.path.join(root, x[2:]), paths)))
    return paths

p=32

def feature_extractor(img, gt, measure, fns, loader=imread):
    ftrs = None
    img = loader(img)
    img = resize(img, (960, 1024, img.shape[-1]))
    for fn in fns:
        def f(blck, axis):
            a = fn(blck, axis=axis)
            d = np.stack([measure(a, gt[0]), measure(a, gt[1])], axis=-1)
            return d
        ftr = block_reduce(img, block_size=(p,p,1), func=f)
        if ftrs is None:
            ftrs = ftr
        else:
            ftrs = np.concatenate((ftrs, ftr), axis=-1)
    return ftrs

# def __mp_f_ext__(x):
#     return

def extract_features(images, gts, measure, fns, n_jobs=1, loader=imread):

    if n_jobs != 1:
        l = list(map(lambda x: (x[0], x[1], measure, fns, loader), zip(images, gts)))
        if n_jobs < 0:
            n_jobs = cpu_count()
        p = pp.Pool(n_jobs)
        features = p.map(lambda x: feature_extractor(x[0], x[1], x[2], x[3], x[4]), l)
        p.close()
        p.join()
        return features
    else:
        features = []
        for img, gt in zip(images, gts):
            ftrs = feature_extractor(img, gt, measure, fns)
            features.append(ftrs)
        features = np.stack(features)
        return features


def img_to_features(images, gts, fns=[np.mean, np.max, np.median], n_jobs=1):
    def load_img(img):
        if type(img) is str:
            img = imread(img)
        return img

    features = extract_features(images, gts.reshape((-1, 2, 3)), lambda x, y: np.linalg.norm(x - y, axis=-1), fns,
                                n_jobs=n_jobs) #2 * 3
    features_cos = extract_features(images, gts.reshape((-1, 2, 3)), cosine_similarity, fns, n_jobs=n_jobs) #2 * 3
    features_com = np.concatenate([features, features_cos], axis=-1)

    gts_uv = to_uv_np(gts.reshape(-1, 2, 3))[0]

    def load_img_uv(img):
        img = load_img(img)
        images_uv, img_Iy = to_uv_np(img)
        return images_uv

    def load_img_y(img):
        img = load_img(img)
        images_uv, img_Iy = to_uv_np(img)
        return img_Iy

    features_ill = np.array([block_reduce(resize(load_img_y(img), (960, 1024)), block_size=(p, p), func=np.mean) for img in images]) #1
    features_uv = extract_features(images, gts_uv, lambda x, y: np.linalg.norm(x - y, ord=1, axis=-1), fns, n_jobs=n_jobs, loader=load_img_uv) #2
    uv = extract_features(images, gts_uv, lambda x, y: x, fns, n_jobs=n_jobs, loader=load_img_uv) #2
    uv = np.array(uv)
    print(uv.shape)
    features_uv = np.concatenate([features_uv, uv[...,0], np.expand_dims(features_ill, -1)], axis=-1)
    # features_uv = np.concatenate([features_uv, np.expand_dims(features_ill, -1)], axis=-1)
    print(features_uv.shape)
    # return features_uv

    features_com1 = np.concatenate([features_com, features_uv], axis=-1)

    # from skimage.color import rgb2lab
    #
    # images_ab = [rgb2lab(img)[..., 1:3] for img in images]
    # gts_ab = rgb2lab(gts.reshape(-1, 2 ,3))[..., 1:3]
    # features_ab = extract_features(images_ab, gts_ab, lambda x, y: np.linalg.norm(x-y, axis=-1), fns)
    # images_ab = None
    # l = np.clip(features_com1[..., -1:], 0, 0.02) * 50
    #
    # features_com12 = features_com1[..., :-1] * l
    # features_com13 = features_ab * l
    # features_com1 = np.concatenate([features_com1[...,-1:], l, features_com12, features_com13], axis=-1)

    return features_com1



def create_target(mask):
    def f(mask, axis):
        mask = np.mean(mask, axis=axis)
        mask = np.argmax(mask, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        return mask
    if type(mask) is str:
        mask = imread(mask)[...,:3]
        if mask.max != 255:
            mask = mask.max() - mask
            mask = mask * 255
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
            mask = np.tile(mask, (1, 1, 3))
    mask = img_as_float(mask)
    m = np.where(mask != np.ones(3), np.array([0., 1., 0.]), mask)  # BLACK CLASS
    m = np.where(mask == np.zeros(3), np.array([0., 0., 1.]), m)
    mask = np.where(mask == np.ones(3), np.array([1., 0., 0.]), m)
    mask = resize(mask, (960, 1024, 3), anti_aliasing=False, order=0)
    mask = block_reduce(mask, block_size=(p,p,1), func=f)
    return mask


def target_creator(masks, n_jobs=1):
    """
    Creates a segmentation target with 3 classes: 0 - one illuminant, 1 - black, 2 - second illuminant.
    The classes are one hot encoded.

    :param masks: Input mask to encode where 0 is the second illuminant, 255 is the first illuminant and 128 is black.
    :return: Encoded mask used for segmentation
    """
    if n_jobs != 1:
        if n_jobs < 0:
            n_jobs = cpu_count()
        p = pp.Pool(n_jobs)
        nm = p.map(create_target, masks)
        p.close()
        p.join()
        return nm
    else:
        nm = []
        for mask in masks:
            mask = create_target(mask)
            nm.append(mask)
        nm = np.stack(nm)
        return nm
