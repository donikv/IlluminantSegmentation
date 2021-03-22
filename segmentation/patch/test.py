from loader import *
from sklearn.preprocessing import *
from general.utils.visualizer import *
from skimage.io import *
import os

import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import load
import multiprocessing

N_JOBS = multiprocessing.cpu_count() - 1

from general.training import metrics
from general.utils import report

if __name__ == '__main__':


    ap = argparse.ArgumentParser(description='Train rf classifier')

    ap.add_argument("--dataset_path", required=True,
                help="Path to the dataset folder")

    ap.add_argument("--dataset_list", required=True,
                help="List of dataset images to use")

    ap.add_argument("--model_path", required=True,
                help="Path to the saved model")

    ap.add_argument("--scaler_path", required=False, default='',
                help="Path to the saved scaler")

    args = vars(ap.parse_args())

    dataset_path = args['dataset_path']
    model_path = args['model_path']
    scaler_path = args['scaler_path']
    dataset_list = args['dataset_list']


    paths = load_paths(dataset_path, dataset_list)
    paths, paths_test = train_test_split(paths, train_size=0.8, test_size=0.2, random_state=42)
    # paths_test = list(filter(lambda x: x.find('outdoor6') != -1 or (x.find('outdoor3')!= -1 and x.find('canon') != -1), paths_test))
    # shuffle(paths)
    # paths_test = paths_test[:10]
    # paths = paths[:100]

    img_paths = list(map(lambda x: x + '/img.png', paths))
    mask_paths = list(map(lambda x: x + '/gt_mask.png', paths))

    images = img_paths
    gts = np.array(list(map(lambda x: np.loadtxt(x + '/gt.txt'), paths)))
    masks = mask_paths
    print("Extracting features...")

    # %%

    img_paths_test = list(map(lambda x: x + '/img.png', paths_test))
    mask_paths_test = list(map(lambda x: x + '/gt_mask.png', paths_test))

    images_test = img_paths_test
    gts_test = np.array(list(map(lambda x: np.loadtxt(x + '/gt.txt'), paths_test)))
    masks_test = mask_paths_test

    features_com_t = img_to_features(images_test, gts_test, n_jobs=8)
    num_features = features_com_t.shape[-1]

    enc = OneHotEncoder(sparse=False, categories=[np.array([0,1,2])])

    X_t = features_com_t.reshape((-1, num_features))
    y_t = target_creator(masks_test)
    y_enc_t = enc.fit_transform(y_t.reshape((-1, 1)))

    # %%

    clf = load(model_path)
    try:
        scaler = load(scaler_path)
        X_t = scaler.transform(X_t)
    except:
        pass

    y_t_pred = clf.predict(X_t)

    # %%
    def encode(mask):
        if mask.max != 255:
            mask = mask.max() - mask
            mask = mask * 255
        mask = img_as_float(mask)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
            mask = np.tile(mask, (1, 1, 3))
        m = np.where(mask != np.ones(3), np.array([0., 1., 0.]), mask)  # BLACK CLASS
        m = np.where(mask == np.zeros(3), np.array([0., 0., 1.]), m)
        mask = np.where(mask == np.ones(3), np.array([1., 0., 0.]), m)
        return mask

    print(y_t_pred)
    print(classification_report(y_t.reshape((-1,)), y_t_pred))
    y_t_pred_e = enc.transform(y_t_pred.reshape((-1, 1)))
    masks_pred = y_t_pred_e.reshape((len(masks_test), 30, 32, 3))
    # mp = list(map(lambda x: imread(x), masks_test))

    masks_pred_full = masks_pred
    masks_true_full = y_enc_t.reshape((len(masks_test), 30, 32, 3))
    dices = []
    it = list(map(lambda x: imread(x), images_test))
    for im, m, pred_m in zip(it, masks_test, masks_pred_full):
        m = encode(imread(m)[...,:3])
        pred_m = resize(pred_m, m.shape, anti_aliasing=False, order=0)
        visualize((im, m, pred_m))
        dice = metrics.dice_coef(m, pred_m, ignore_black=True)
        dice = np.mean(dice)
        dices.append(dice)

    stat = report.error_statistics(dices)
    report.print_report(stat)


# %%

