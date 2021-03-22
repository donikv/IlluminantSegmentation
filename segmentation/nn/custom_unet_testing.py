import tensorflow as tf
from general.training import custom_training
from general.processing import data_processing as dp
from general.utils import visualizer
import os
from general.training.metrics import dice_coef
from general.utils.report import write_report, error_statistics
from general.datasets.Cube2Dataset import encode_mask

import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Relight images with two illuminants')

    ap.add_argument("--config_folder", required=True,
                    help="Path to the folder containing the configuration file")
    ap.add_argument("--training_instance", required=True,
                    help="Name of the config file")

    args = vars(ap.parse_args())

    config_folder = args['config_folder']
    date = args["training_instance"]

config_folder = 'training/outdoor_fpn_small_clustering'
date = "20210319-1035"

save_dir = f'{config_folder}/model/{date}/model.ckpt'
resultsf = f'{config_folder}/results/{date}/dices.txt'
os.makedirs(os.path.dirname(resultsf), exist_ok=True)
f = open(resultsf, 'a+')

training_config = custom_training.CustomTraining(f'{config_folder}/training.config')
ds, count = training_config.get_dataset_from_config(type=dp.TEST)

params = training_config.get_fit_parameters_from_config()

model = training_config.load_model_from_config()
model.load_weights(save_dir)
model.compile(**params)

training_params = training_config.get_training_parameters()
EPOCHS = training_params.epochs
BS = training_params.batch_size
STEPS_PER_EPOCH = count // BS


def plot_prediction(img, mask, pred_mask):
    mask = mask[0]
    pred_mask = pred_mask[0]
    if pred_mask.shape[-1] > 3:
        visualizer.visualize([img[0, :, :, :3], img[0, :, :, 3:6], mask[..., :3],
                              tf.squeeze(pred_mask[..., :-3]), mask[..., 3:], pred_mask[..., -3:]])
    else:
        visualizer.visualize([img[0, :, :, :3], img[0, :, :, 3:6], mask[..., :3], tf.squeeze(pred_mask[..., :3])])


class DisplayCallback(tf.keras.callbacks.Callback):

    def __init__(self, ds):
        super().__init__()
        self.images = list(ds.take(3))

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 4 != 0:
            return
        for data in self.images:
            img, mask = data[0], data[1]
            pred_mask = model.predict(img)
            plot_prediction(img, mask, pred_mask)
        print('/nSample Prediction after epoch {}\n'.format(epoch + 1))

dices = []
dices_ib = []
dices_ib_inv = []
for data in iter(ds):
    image, mask = data[0], data[1]
    Y = model(image, training=False)
    if type(mask) == tuple:
        mask = mask[0]
    if type(Y) == tuple or type(Y) == list:
        Y = Y[0]
    if Y.shape[-1] % 3 != 0:
        Y = tf.tile(Y[..., :1], (1, 1, 1, 3))
        Y = tf.round(Y)
        mask = tf.round(mask)
        Y = encode_mask(Y, image[..., :3])
        mask = encode_mask(mask[..., :1], image[..., :3])

    diceib = dice_coef(Y, mask, ignore_black=True)
    diceib = tf.reduce_mean(diceib)
    dices_ib.append(diceib)

    diceib_inv = tf.maximum(diceib, 1-diceib)
    dices_ib_inv.append(diceib_inv)

    dice = dice_coef(Y, mask, ignore_black=False)
    dice = tf.reduce_mean(dice)
    dices.append(dice)

    plot_prediction(img=image, mask=mask, pred_mask=Y)

dices_stat = error_statistics(dices)
dices_ib_stat = error_statistics(dices_ib)
dices_ib_inv_stat = error_statistics(dices_ib_inv)

write_report(dices_stat, f)
write_report(dices_ib_stat, f)
write_report(dices_ib_inv_stat, f)
f.flush()
f.close()