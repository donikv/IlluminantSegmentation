import tensorflow as tf
from general.training import custom_training
from general.processing import data_processing as dp
from general.utils import visualizer
import os
from general.training.metrics import dice_coef, cos_sim
from general.utils.report import write_report, error_statistics
from general.datasets.Cube2Dataset import encode_mask
import numpy as np
import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Relight images with two illuminants')

    ap.add_argument("--config_folder", required=True,
                    help="Path to the folder containing the configuration file")
    ap.add_argument("--config_name", required=True,
                    help="Name of the config file")
    ap.add_argument("--training_instance", required=True,
                    help="Name of the config file")

    args = vars(ap.parse_args())

    config_folder = args['config_folder']
    config_name = args["config_name"]
    date = args["training_instance"]


    save_dir = f'{config_folder}/model/{date}/model.ckpt'
    resultsf = f'{config_folder}/results/{date}/dices.txt'
    os.makedirs(os.path.dirname(resultsf), exist_ok=True)
    f = open(resultsf, 'a+')

    training_config = custom_training.CustomTraining(f'{config_folder}/{config_name}.config')
    ds, count = training_config.get_dataset_from_config(type=dp.TEST)

    params = training_config.get_fit_parameters_from_config()

    model = training_config.load_model_from_config()
    model.load_weights(save_dir)
    model.compile(**params)

    training_params = training_config.get_training_parameters()
    EPOCHS = training_params.epochs
    BS = training_params.batch_size
    STEPS_PER_EPOCH = count // BS

    def correct(img, mask, gt1, gt2):
        img = tf.where(tf.reduce_max(img, axis=-1, keepdims=True) > 0.95 * tf.reduce_max(img), tf.zeros_like(img), img)
        gt_mask = (1-mask) * gt1 + mask * gt2
        img = img / gt_mask
        return img, gt_mask

    def create_mask(mask, gt1, gt2):
        gt_mask = (1 - mask) * gt1 + mask * gt2
        return gt_mask

    def gw_prediction(img, mask):
        img1 = img[..., :3] * mask
        gt = tf.reduce_mean(img1, axis=[-3,-2])
        return gt


    def plot_prediction(img, mask, pred_mask):
        gt1, gt2 = None, None
        if type(mask) == tuple:
            gt1 = tf.cast(mask[-2], tf.float32)
            gt2 = tf.cast(mask[-1], tf.float32)
            pgt1 = pred_mask[-2]
            pgt2 = pred_mask[-1]
            m = mask[-3]
            y = pred_mask[-3]
            if len(mask) == 6:
                m += 1 - mask[-4]
                m /= 2
                y += 1 - pred_mask[-4]
                y /= 2
            pred_mask = y
            mask = m
        mask = mask[0]
        pred_mask = pred_mask[0]
        if gt1 is not None:
            corrected, gt_mask = correct(img[0, :, :, :3], pred_mask[..., :3], pgt1[0], pgt2[0])
            corrected1, gt_mask1 = correct(img[0, :, :, :3], mask[..., :3], gt1[0], gt2[0])
            corrected11, _ = correct(img[0, :, :, :3], mask[..., :3], pgt1[0], pgt1[0])
            corrected12, _ = correct(img[0, :, :, :3], mask[..., :3], pgt2[0], pgt2[0])
            # visualizer.visualize(img[:, :, :, :3])
            # visualizer.visualize([corrected1, corrected])
            # visualizer.visualize([corrected11, corrected12])
            # visualizer.visualize([gt_mask1, gt_mask])
            # visualizer.visualize([tf.squeeze(mask[..., :3]), tf.squeeze(pred_mask[..., :3])], cb=False)
            if gt1.shape[-1] == 2:
                gt1 = tf.stack([gt1[..., 0], tf.ones_like(gt1[..., 0]), gt1[..., 1]], axis=-1)
                pgt1 = tf.stack([pgt1[..., 0], tf.ones_like(pgt1[..., 0]), pgt1[..., 1]], axis=-1)
                gt2 = tf.stack([gt2[..., 0], tf.ones_like(gt2[..., 0]), gt2[..., 1]], axis=-1)
                pgt2 = tf.stack([pgt2[..., 0], tf.ones_like(pgt2[..., 0]), pgt2[..., 1]], axis=-1)
            gt1 = visualizer.create_mask(gt1[0], (10, 10))
            gt2 = visualizer.create_mask(gt2[0], (10, 10))
            pgt1 = visualizer.create_mask(pgt1[0], (10, 10))
            pgt2 = visualizer.create_mask(pgt2[0], (10, 10))
        if gt1 is not None:
            visualizer.visualize([img[0, :, :, :3], img[0, :, :, 3:6], tf.squeeze(mask[..., :3]),
                                  tf.squeeze(pred_mask[..., :3]), gt1, gt2, pgt1, pgt2, corrected, gt_mask])
        else:
            visualizer.visualize(
                [img[0, :, :, :3], img[0, :, :, 3:6], tf.squeeze(mask[..., :3]), tf.squeeze(pred_mask[..., :3])])

    dices = []
    cos1 = []
    cos2 = []
    cos_mask = []
    rgbg = []
    use_intermediate = False
    for data in iter(ds):
        image, mask = data[0], data[1]
        # t0 = time.time_ns()
        Y = model(image, training=False)
        # t1 = time.time_ns()
        # print((t1-t0)/1000000)
        if not use_intermediate:
            plot_prediction(img=image, mask=mask, pred_mask=Y)
        if type(mask) == tuple:
            gt1 = tf.cast(mask[-2], tf.float32)
            gt2 = tf.cast(mask[-1], tf.float32)
            pgt1 = Y[-2]
            pgt2 = Y[-1]
            m = mask[-3]
            y = Y[-3]
            if len(mask) == 6:
                m += 1 - mask[-4]
                m /= 2
                y += 1 - Y[-4]
                y /= 2
            Y=y
            mask=m
        if use_intermediate:
            fn = tf.keras.backend.function
            pgt1 = tf.convert_to_tensor(fn([model.input], [model.get_layer("estimation31").output])(image)[0])
            pgt2 = tf.convert_to_tensor(fn([model.input], [model.get_layer("estimation32").output])(image)[0])
            Y = tf.convert_to_tensor(fn([model.input], [model.get_layer("segmentation4").output])(image)[0])

        # pgt1 = gw_prediction(image, 1-mask)
        # pgt2 = gw_prediction(image, mask)

        gt_mask = create_mask(mask[..., :3], gt1[0], gt2[0])
        pgt_mask = create_mask(Y[..., :3], pgt1[0], pgt2[0])
        cos_mask.append(tf.reduce_mean(cos_sim(gt_mask, pgt_mask)))

        if Y.shape[-1] % 3 != 0:
            Y = tf.tile(Y[..., :1], (1, 1, 1, 3))
            Y = tf.round(Y)
            mask = tf.round(mask)
            Y = encode_mask(Y, image[..., :3])
            mask = encode_mask(mask[..., :1], image[..., :3])

        dice = dice_coef(Y, mask, ignore_black=False)
        dice = tf.reduce_mean(dice)
        dices.append(dice)

        cos1.append(tf.reduce_mean(cos_sim(y_true=gt1, y_pred=pgt1)))
        cos2.append(tf.reduce_mean(cos_sim(y_true=gt2, y_pred=pgt2)))

        rgbg.append((gt1[0, 0::2] / gt1[0, 1]).numpy())
        rgbg.append((gt2[0, 0::2] / gt2[0, 1]).numpy())
        rgbg.append((pgt1[0, 0::2] / pgt1[0, 1]).numpy())
        rgbg.append((pgt2[0, 0::2] / pgt2[0, 1]).numpy())

    rgbg = np.array(rgbg)


    dices_stat = error_statistics(dices)
    cos1_stat = error_statistics(cos1)
    cos2_stat = error_statistics(cos2)
    cosm_stat = error_statistics(cos_mask)

    write_report(dices_stat, f)
    write_report(cos1_stat, f)
    write_report(cos2_stat, f)
    write_report(cosm_stat, f)
    f.flush()
    f.close()

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(rgbg[0::4, 0], rgbg[0::4, 1], label='True shadow', s=3)
    ax.scatter(rgbg[1::4, 0], rgbg[1::4, 1], label='True sun', s=3)
    ax.scatter(rgbg[2::4, 0], rgbg[2::4, 1], label='Predicted shadow', s=3)
    ax.scatter(rgbg[3::4, 0], rgbg[3::4, 1], label='Predicted sun', s=3)
    plt.legend(loc='upper right')
    plt.show()