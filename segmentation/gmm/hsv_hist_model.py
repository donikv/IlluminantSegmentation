from general.datasets.Cube2Dataset import encode_mask
from general.processing.data_processing import *
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans
import math
from general.processing import histogram


def fit_model(image, gt=None, n_classes=2, mixture='gmm',hist_smoothing_f=lambda x:x, bn_cnt=256, mode='hue', w=True):
    if len(image.shape) == 3:
        image = tf.expand_dims(image, axis=0)

    if gt is not None:
        gt = tf.stack((gt[..., :3], gt[..., 3:]), axis=1)
        gt_hsv = tf.image.rgb_to_hsv(gt)
        gt_h = gt_hsv.numpy()[..., 0].reshape(-1,1)
    else:
        gt_h = None

    ihsv = tf.image.rgb_to_hsv(image)

    _, Iy = histogram.to_uv_np(image)
    Iy_flat = Iy.ravel()
    ishv_flat = ihsv.numpy()[..., 0].ravel()

    tresh = Iy_flat.max() / 2 if Iy_flat.max() < 0.005 else 0.005
    if mode == 'hue':
        ishv_flat_filtered = ishv_flat[Iy_flat > tresh]
        Iy_flat_filt = Iy_flat[Iy_flat > tresh]

    else:
        Iy_flat_filt = Iy_flat[Iy_flat > tresh]
        ishv_flat_filtered = ishv_flat[Iy_flat > tresh]

    hist, idexs = np.histogram(ishv_flat_filtered, range=[0, 1], bins=bn_cnt, weights=Iy_flat_filt)
    hist1 = np.clip(hist_smoothing_f(hist), 0, np.inf).astype(int)
    hist1 = np.where(hist1 < 5, np.zeros_like(hist1), hist1)
    weighted = np.repeat(idexs[1:], hist1)
    if len(weighted) < 1 or not w:
        weighted = ishv_flat_filtered

    if mixture == 'gmm':
        gmm = GaussianMixture(n_components=n_classes, max_iter=1000, means_init=gt_h)
    elif mixture == 'kmeans':
        gmm = KMeans(n_clusters=n_classes)
    else:
        gmm = BayesianGaussianMixture(n_components=n_classes, max_iter=1000, weight_concentration_prior_type="dirichlet_distribution")
    gmm.fit((weighted if mode == 'hue' else Iy_flat_filt).reshape(-1, 1))

    return gmm


def fit_uv_hist_model(image, gt=None, n_classes=2, mixture='gmm', hist_smoothing_f=lambda x:x, bn_cnt=256, eps=histogram.EPS):
    if len(image.shape) == 3:
        image = tf.expand_dims(image, axis=0)

    if gt is not None:
        gt = tf.stack((gt[..., :3], gt[..., 3:]), axis=1)
        gt_hsv = tf.image.rgb_to_hsv(gt)
        gt_h = gt_hsv.numpy()[..., 0].reshape(-1,1)
    else:
        gt_h = None

    iuv, Iy = histogram.to_uv_np(image)
    Iy_flat = Iy.ravel()
    iuv = iuv.reshape((-1, 2))

    iuv = iuv[Iy_flat > 0.02]
    Iy_flat = Iy_flat[Iy_flat > 0.02]

    rng = (iuv.min(), iuv.max())
    hist, idexs = np.histogramdd(iuv, range=[rng, rng], bins=bn_cnt, weights=Iy_flat)
    hist1 = np.clip(hist_smoothing_f(hist), 0, np.inf).astype(int)
    mesh = np.meshgrid(idexs[0][1:], idexs[1][1:])
    mesh = np.stack(mesh, axis=-1).reshape((-1, 2))
    weighted = np.repeat(mesh, hist1.reshape((-1,)), axis=0)

    if mixture == 'gmm':
        gmm = GaussianMixture(n_components=n_classes, max_iter=1000, means_init=gt_h)
    else:
        gmm = BayesianGaussianMixture(n_components=n_classes, max_iter=1000)
    gmm.fit(weighted)

    return gmm



def get_prediction(gmm, image, Iy=None, mode='hue', n_classes=2, encode=True, return_all=True):
    ihsv = tf.image.rgb_to_hsv(image)

    b, h, w, c = image.shape

    if Iy is None:
        iuv, Iy = histogram.to_uv_np(image)
    else:
        iuv, _ = histogram.to_uv_np(image)

    Iy_flat = Iy.ravel()
    ishv_flat = ihsv.numpy()[..., 0].ravel()

    if mode == 'hue':
        p = ishv_flat.reshape(-1, 1)
    elif mode == 'lum':
        p = Iy_flat.reshape(-1, 1)
    elif mode == 'uv':
        p = iuv.reshape((-1, 2))


    Y = gmm.predict(p)
    if n_classes > 2 and encode:
        Y = np.where(Y < math.ceil(n_classes / 2), np.zeros_like(Y), np.ones_like(Y))
    # Y = np.where(Iy_flat < 0.02, np.ones_like(Y) * 0.5, Y)
    Y = Y.reshape((h, w, 1))
    if encode:
        Y_t = tf.cast(tf.tile(Y, (1, 1, 3)), float)
        Y_enc = encode_mask(Y_t, image[0])
    else:
        Y = Y.squeeze()

    if return_all:
        return Y_enc if encode else Y, Iy_flat, p, ihsv
    else:
        return Y_enc if encode else Y