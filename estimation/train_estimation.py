import tensorflow as tf
from general.training import custom_training
import datetime
from general.processing import data_processing as dp

config_folder = 'training/simplefc4_gw'
config_name = "training_tau_csn"


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
                          callbacks=[cp_callback, cp_callback_all, stp, rate, tensorboard_callback])
