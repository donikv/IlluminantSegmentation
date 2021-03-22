import tensorflow as tf
from general.training import custom_training
from general.processing import data_processing as dp
import os
from general.training.metrics import cos_sim
from general.utils.report import write_report, error_statistics, print_report
import numpy as np


config_folder = 'training/simplefc4_gw/'
date = "20210211-1025_training_tau_csn"
config_name = "training_tau_csn"

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

cos1 = []
rgbg = []
for data in iter(ds):
    image, mask = data[0], data[1]
    mask = tf.cast(mask, tf.float32)
    Y = model(image)

    cos1.append(tf.reduce_mean(cos_sim(y_true=mask, y_pred=Y)))
    rgbg.append((mask[0, 0::2] / mask[0,1]).numpy())
    rgbg.append((Y[0, 0::2] / Y[0,1]).numpy())

rgbg = np.array(rgbg)

cos1_stat = error_statistics(cos1)

print_report(cos1_stat)
write_report(cos1_stat, f)

f.flush()
f.close()

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(rgbg[0::2, 0], rgbg[0::2, 1], label='true', s=3)
ax.scatter(rgbg[1::2, 0], rgbg[1::2, 1], label='predicted', s=3)
plt.legend(loc='upper right')
plt.show()