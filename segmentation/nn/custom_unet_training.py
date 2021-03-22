import tensorflow as tf
from general.training import custom_training
import datetime
from general.processing import data_processing as dp
from general.utils import visualizer
import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Relight images with two illuminants')

    ap.add_argument("--config_folder", required=True,
                    help="Path to the folder containing the configuration file")

    args = vars(ap.parse_args())

    config_folder = args['config_folder']

    log_dir = f'{config_folder}/logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M")}'
    save_dir = f'{config_folder}/model/{datetime.datetime.now().strftime("%Y%m%d-%H%M")}/model.ckpt'

    training_config = custom_training.CustomTraining(f'{config_folder}/training.config')
    ds_train, count = training_config.get_dataset_from_config(type=dp.TRAIN)
    ds_valid, count_valid = training_config.get_dataset_from_config(type=dp.VALID)

    params = training_config.get_fit_parameters_from_config()

    model = training_config.load_model_from_config()
    model.compile(**params)

    training_params = training_config.get_training_parameters()
    EPOCHS = training_params.epochs
    BS = training_params.batch_size
    STEPS_PER_EPOCH = count // BS
    VALIDATION_STEPS = count_valid // BS


    class DisplayCallback(tf.keras.callbacks.Callback):

        def __init__(self, ds):
            super().__init__()
            if "Dataset" in str(type(ds)):
                self.images = list(ds.take(3))
            else:
                self.images = [next(ds), next(ds), next(ds)]

        def on_epoch_end(self, epoch, logs=None):
            if epoch % 4 != 0:
                return
            for data in self.images:
                img, mask = data[0], data[1]
                pred_mask = model.predict(img)
                if type(mask) == tuple:
                    mask = mask[0]
                if type(pred_mask) == tuple:
                    pred_mask = pred_mask[0]
                mask = mask[0]
                pred_mask = pred_mask[0]
                # mask = histogram.from_uv(histogram.decode_bins(mask[0], depth))
                # pred_mask = histogram.from_uv(histogram.decode_bins(pred_mask[0], depth))
                if len(mask.shape) == 1:
                    mask = visualizer.create_mask(mask, (10, 10))
                    mask = tf.tile(mask, (1, 1, 2))
                if pred_mask.shape[-1] > 3:
                    visualizer.visualize([img[0, :, :, :3], img[0, :, :, 3:6], tf.squeeze(mask[..., :-3]),
                                          tf.squeeze(pred_mask[...,:-3]), mask[...,-3:], pred_mask[...,-3:]])
                else:
                    visualizer.visualize([img[0, :, :, :3], img[0, :, :, 3:6], tf.squeeze(mask[..., :3]), tf.squeeze(pred_mask[..., :3])])

            print('/nSample Prediction after epoch {}\n'.format(epoch + 1))

    dc = DisplayCallback(ds_valid)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_dir,
                                                     monitor='val_loss',
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     verbose=1)

    cp_callback_all = tf.keras.callbacks.ModelCheckpoint(filepath=save_dir + 'x',
                                                         monitor='val_loss',
                                                         save_weights_only=True,
                                                         save_best_only=False,
                                                         verbose=1)

    rate = tf.keras.callbacks.ReduceLROnPlateau(patience=50, verbose=1)

    stp = tf.keras.callbacks.EarlyStopping("val_loss", mode="min", patience=400)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model_history = model.fit(ds_train, epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=ds_valid,
                              callbacks=[cp_callback, cp_callback_all, stp, rate, dc, tensorboard_callback])
