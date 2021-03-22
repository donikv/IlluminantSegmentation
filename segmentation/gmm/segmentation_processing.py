import tensorflow as tf


#Segementation features


@tf.function
def __img_to_ill_diff__(image, gt):
    h, w, c = image.shape
    image = filter_img(image, gaussian_kernel(5))[0]
    iuv, Iy = histogram.to_uv(image)
    gt1, gt2 = gt[0], gt[1]
    gt1 = tf.tile(gt1[tf.newaxis, tf.newaxis, :], (h, w, 1))
    gt1_uv, _ = histogram.to_uv(gt1)
    diff1 = tf.linalg.norm(iuv - gt1_uv, axis=-1)
    diff1 = tf.where(Iy < 0.02, tf.zeros_like(diff1, dtype=tf.float32), diff1)

    gt2 = tf.tile(gt2[tf.newaxis, tf.newaxis, :], (h, w, 1))
    gt2_uv, _ = histogram.to_uv(gt2)
    # diff2 = cosine_similarity(gt2, image)
    diff2 = tf.linalg.norm(iuv - gt2_uv, axis=-1)
    diff2 = tf.where(Iy < 0.02, tf.zeros_like(diff2), diff2)

    # diff1 = tf.math.divide_no_nan(diff1, tf.reduce_mean(diff1))
    # diff2 = tf.math.divide_no_nan(diff2, tf.reduce_mean(diff2))

    conf = tf.abs((diff1 - diff2)) / (diff1 + diff2 + 1e-7)
    conf = conf / tf.reduce_max(conf)
    cnf2 = -tf.math.log(1 - conf)
    conf2 = tf.where(tf.math.is_nan(cnf2), tf.zeros_like(cnf2), cnf2)

    diff1c = diff1 * cnf2
    diff2c = diff2 * cnf2
    # diff1c = diff1c / tf.reduce_max(diff1c)
    # diff2c = diff2c / tf.reduce_max(diff2c)

    # m1 = tf.where(diff1c > diff2c, tf.ones_like(diff1c, dtype=tf.uint8) * 2, tf.zeros_like(diff1, dtype=tf.uint8))
    # m1 = tf.where(diff1c == diff2c, tf.ones_like(diff1c, dtype=tf.uint8), m1)
    # m1 = tf.one_hot(m1, depth=3)
    # return m1
    d = tf.stack([diff1, diff2, conf], axis=-1)
    d = tf.where(tf.math.is_nan(d), tf.zeros_like(d), d)
    tf.debugging.assert_all_finite(
        d, 'Features must not contain nan values', name=None
    )
    return d
    # return tf.concat((d, m1), axis=-1)


@tf.function
def segmenation_features_dataset(*x, gamma=2, gain=2):
    image = tf.image.adjust_gamma(x[0], gamma=gamma, gain=gain)
    gt = tf.convert_to_tensor(x[-1])
    gt = tf.stack((gt[:3], gt[3:]), axis=0) if len(gt.shape) == 1 else gt
    f = __img_to_ill_diff__(image, gt)
    f = tf.concat((f, image), axis=-1)
    return (f, *x[1:], x[0])


def hsv_dataset(*x):
    hsv = tf.image.rgb_to_hsv(x[0])
    return (hsv, *x[1:], x[0])


def plot_prediction(h, weights, bn_cnt, gmm, f, Y_enc):
    gmm_x = np.arange(h.min(), h.max(), (h.max() - h.min())/bn_cnt)
    if h.shape[-1] == 2:
        mesh = np.meshgrid(gmm_x, gmm_x)
        gmm_x = np.stack(mesh, axis=-1)

    c = 2 if h.shape[-1] == 2 else 1
    gmm_y = gmm.score_samples(gmm_x.reshape(-1, c))
    gmm_y = np.exp(gmm_y)
    fig, ax = plt.subplots()
    if h.shape[-1] == 2:
        gmm_x1 = (gmm_x - h.min()) * bn_cnt / (h.max() - h.min())
        gmm_x1 = np.flip(gmm_x1, axis=-1)
        rng = (h.min(), h.max())
        hist, idexs = np.histogramdd(h, range=[rng, rng], bins=bn_cnt, weights=f(weights))
        ax.imshow(hist)
        ax.contour(gmm_x1[..., 0], gmm_x1[..., 1], gmm_y.reshape(gmm_x[...,0].shape), 8, colors='w')
    else:
        ax.hist(h, bins=bn_cnt, range=[0, h.max()], density=True, weights=f(weights))
        ax.plot(gmm_x, gmm_y, color="red")
    plt.show()
    # visualizer.visualize([features[0, :, :, 0], features[0, :, :, 1], features[0, :, :, 2], image[0], mask[0]])
    visualizer.visualize([Y_enc, mask[0]])


if __name__ == '__main__':
    from general.datasets.Cube2Dataset import load_paths, encode_mask
    from general.processing.data_processing import *
    import matplotlib.pyplot as plt
    from general.training.metrics import dice_coef
    from general.utils import report, visualizer
    import hsv_hist_model
    from general.processing import data_processing as dp

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    bs = 1

    paths = load_paths('/media/donik/Disk/Cube2', 'list.txt')
    # shuffle(paths)
    # paths = np.array(list(filter(lambda x: x.find('indoor') != -1, paths)))
    valid_paths = np.array(list(filter(lambda x: x.find('S_') != -1, paths)))

    # ds_base = dataset(paths, type=TEST, bs=bs, cache=False, regression=False, gt=True,
    #                   map_fn=segmenation_features_dataset, shuffle_buffer_size=1)
    # ds = ds_base
    dices_l = []
    dices_h = []
    dices_max = []
    plot = False
    for path in paths:
        img = Cube2Dataset.get_image(path, 'img.png', Cube2Dataset.IMG_HEIGHT // 2, Cube2Dataset.IMG_WIDTH // 2)
        mask = encode_mask(Cube2Dataset.get_image(path, 'gt_mask_round.png', Cube2Dataset.IMG_HEIGHT // 2, Cube2Dataset.IMG_WIDTH // 2), img)
        gt = Cube2Dataset.get_gt(path)
        # data = segmenation_features_dataset(img, mask, gt, gamma=1, gain=1)
        image = tf.expand_dims(img, axis=0)
        mask = tf.expand_dims(mask, axis=0)
        # f = np.sqrt
        # visualizer.visualize([features[0,:,:,0], features[0,:,:,1], features[0,:,:,2], features[0,:,:,3:]])
        # visualizer.visualize([image[0], mask[0]])
        bn_cnt = 1000
        # image_gamma = tf.image.adjust_gamma(image, 2, gain=4)
        _, Iy = histogram.to_uv_np(image)
        # image_der = dp.__process_images__(image, [3])

        # Y = hsv_hist_model.model(image, encode=False, gt=None, n_classes=3, mixture='bgmm', hist_smoothing_f=f, bn_cnt=bn_cnt, return_gmm=False)
        image_blur = dp.filter_img(image, dp.gaussian_kernel(5))
        if plot:
            visualizer.visualize(image_blur)
        gmm = hsv_hist_model.fit_model(image_blur, gt=None, n_classes=2, mixture='gmm', bn_cnt=bn_cnt, w=False)
        Y_enc, Iy_flat, iuv_flat, ihsv = hsv_hist_model.get_prediction(gmm, image, Iy, n_classes=2, mode='hue')

        gmm_l = hsv_hist_model.fit_model(image, gt=None, n_classes=2, mixture='bgmm', bn_cnt=bn_cnt, mode='lum')
        Y_enc_l, _, _, _ = hsv_hist_model.get_prediction(gmm_l, image, Iy, n_classes=2, mode='lum')

        dice = dice_coef(Y_enc, mask[0], ignore_black=True)
        dice = tf.reduce_mean(dice)
        dice = tf.maximum(dice, 1-dice)
        dices_h.append(dice)
        print(dice)

        dice1 = dice_coef(Y_enc_l, mask[0], ignore_black=True)
        dice1 = tf.reduce_mean(dice1)
        dice1 = tf.maximum(dice1, 1-dice1)
        dices_l.append(dice1)
        print(dice1)
        print()
        dices_max.append(max(dice1, dice))

        # Iy_flat_filt = Iy_flat[ishv_flat > 0.005]
        iuv_flat_filt = iuv_flat[Iy_flat > 0.01]
        Iy_flat_filt = Iy_flat[Iy_flat > 0.01]
        if plot:
            plot_prediction(iuv_flat_filt, Iy_flat_filt, bn_cnt, gmm, lambda x: x, Y_enc)
            plot_prediction(Iy_flat_filt, None, bn_cnt, gmm_l, lambda x: x, Y_enc_l)
            visualizer.visualize([image[0], mask[0]])

        f = open(path + '/dices.txt', 'w+')
        f.write(f'{dice} {dice1}\n')
        f.close()

    stat = report.error_statistics(dices_h)
    report.print_report(stat)
    stat = report.error_statistics(dices_l)
    report.print_report(stat)
    stat = report.error_statistics(dices_max)
    report.print_report(stat)