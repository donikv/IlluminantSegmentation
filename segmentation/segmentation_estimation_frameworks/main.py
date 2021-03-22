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
    ap.add_argument("--config_name", required=True,
                    help="Name of the config file")

    args = vars(ap.parse_args())

    config_folder = args['config_folder']
    config_name = args["config_name"]


    log_dir = f'{config_folder}/logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M")}_{config_name}'
    save_dir = f'{config_folder}/model/{datetime.datetime.now().strftime("%Y%m%d-%H%M")}_{config_name}/model.ckpt'

    training_config = custom_training.CustomTraining(f'{config_folder}/{config_name}.config')
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
                gt1, gt2 = None, None
                if type(mask) == tuple:
                    gt1 = mask[-2]
                    gt2 = mask[-1]
                    mask = mask[-3]
                    pgt1 = pred_mask[-2][...,:3]
                    pgt2 = pred_mask[-1][...,:3]
                    pred_mask = pred_mask[-3][...,:mask.shape[-1]]
                mask = mask[0]
                pred_mask = pred_mask[0]
                if gt1 is not None:
                    if gt1.shape[-1] == 2:
                        gt1 = tf.stack([gt1[...,0], tf.ones_like(gt1[...,0]), gt1[...,1]], axis=-1)
                        pgt1 = tf.stack([pgt1[...,0], tf.ones_like(pgt1[...,0]), pgt1[...,1]], axis=-1)
                        gt2 = tf.stack([gt2[...,0], tf.ones_like(gt2[...,0]), gt2[...,1]], axis=-1)
                        pgt2 = tf.stack([pgt2[...,0], tf.ones_like(pgt2[...,0]), pgt2[...,1]], axis=-1)
                    gt1 = visualizer.create_mask(gt1[0], (10, 10))
                    gt2 = visualizer.create_mask(gt2[0], (10, 10))
                    pgt1 = visualizer.create_mask(pgt1[0], (10, 10))
                    pgt2 = visualizer.create_mask(pgt2[0], (10, 10))
                if gt1 is not None:
                    visualizer.visualize([img[0, :, :, :3], img[0, :, :, 3:6], tf.squeeze(mask[..., :3]),
                                          tf.squeeze(pred_mask[...,:3]), gt1, gt2, pgt1, pgt2])
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

    stp = tf.keras.callbacks.EarlyStopping("val_loss", mode="min", patience=1000)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model_history = model.fit(ds_train, epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=ds_valid,
                              callbacks=[cp_callback, cp_callback_all, stp, rate, dc, tensorboard_callback])
