from general.datasets.NewCube2Dataset import dataset, load_paths, encode_mask
from general.processing.data_processing import *
import matplotlib.pyplot as plt
from general.training.metrics import dice_coef
from general.utils import report, visualizer
import hsv_hist_model
from sklearn.model_selection import train_test_split
import os
import itertools

def load_labels():
    # indoor and outdoor relevant
    file_name_IO = 'data/IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    return labels_IO


def permuted_dice(yenc, mask, ignore_black=True):
    dc = 0.
    cls_perm = itertools.permutations([0, 2], 2)
    pred_mask = yenc
    for perm in cls_perm:
        perm = (perm[0], 1, perm[1])
        yp = yenc.numpy()[:, :, perm]
        ndc = dice_coef(mask, yp, ignore_black=ignore_black).numpy().mean()
        if ndc > dc:
            pred_mask = yp
        dc = np.maximum(ndc, dc)
    return dc, pred_mask

if __name__ == '__main__':

    ap = argparse.ArgumentParser(description='Train rf classifier')

    ap.add_argument("--dataset_path", required=True,
                    help="Path to the dataset folder")

    ap.add_argument("--dataset_list", required=True,
                    help="List of dataset images to use")

    args = vars(ap.parse_args())

    dataset_path = args['dataset_path']
    dataset_list = args['dataset_list']

    bs = 1

    paths = load_paths(dataset_path, dataset_list)
    _, test = train_test_split(paths, train_size=0.8, test_size=0.2, random_state=42)

    tests = [
        test
        # np.array(list(filter(lambda x: x.find('outdoor') != -1, paths))),
        # np.array(list(filter(lambda x: x.find('lab') != -1, paths))),
        # np.array(list(filter(lambda x: x.find('indoor') != -1, paths))),
    ]
    stats1 = []
    stats2 = []
    bn_cnt = 1000

    H, W = 224, 224
    classes = 365
    model = tf.keras.models.load_model("pretrained/places365_model")
    model.compile()
    model.summary()
    labels_IO = load_labels()

    for i, test_paths in enumerate(tests):

        ds_base = dataset(test_paths, type=TEST, bs=bs, cache=False, regression=False, round=False, gt=True, shuffle_buffer_size=1)
        ds = ds_base
        dices = []
        dices2 = []
        plot = False
        plot_res = True
        print(test_paths[0])
        cnt = 0
        correct = 0
        img_idx = 0
        good_plots, bad_plots = 0, 0
        for image, mask, gt in iter(ds):
            img_idx += 1
            gw = tf.reduce_mean(image, axis=[1,2])
            gw = gw / (tf.reduce_max(gw, axis=-1, keepdims=True) + 1e-7)
            img_gw = image / (gw + 1e-7)
            if plot:
                visualizer.visualize(img_gw)
            img_gw = tf.image.per_image_standardization(img_gw)
            img_gw = tf.image.resize(img_gw, (H, W))
            if plot:
                visualizer.visualize(img_gw)

            logits = model.predict(img_gw)
            probs = tf.nn.softmax(logits)
            # plt.plot(np.arange(0, 365, 1), probs[0])
            # plt.show()
            idx = tf.argsort(probs, direction="DESCENDING")
            idx = idx.numpy().squeeze()

            # output the IO prediction
            io_image = np.mean(labels_IO[idx[:10]])  # vote for the indoor or outdoor
            indoor = io_image < 0.5
            correct += 1 if (indoor and i > 0) or (not indoor and i < 1) else 0
            cnt += 1

            image = image * 5
            mask = encode_mask(mask, image)

            _, Iy = histogram.to_uv_np(image)
            # image_gamma = tf.image.adjust_gamma(image, 2, gain=2)
            gmm = hsv_hist_model.fit_model(image, bn_cnt=bn_cnt, gt=None, n_classes=2, hist_smoothing_f=np.sqrt, mixture="bgmm")
            Y_enc, ishv_flat, Iy_flat, _ = hsv_hist_model.get_prediction(gmm, image, Iy, n_classes=2, mode='hue')

            dice = dice_coef(Y_enc, mask[0], ignore_black=True)
            dice = tf.reduce_mean(dice)
            dice = tf.maximum(dice, 1-dice)
            dice, Y_enc = permuted_dice(Y_enc, mask[0], ignore_black=False)
            # print(dice)

            gmm_lum = hsv_hist_model.fit_model(image, gt=None, n_classes=2, mixture='bgmm',
                                                                   bn_cnt=bn_cnt, mode='lum')
            Y_enc_l, _, _, _ = hsv_hist_model.get_prediction(gmm_lum, image, Iy, n_classes=2, mode='lum')

            dice1 = dice_coef(Y_enc_l, mask[0], ignore_black=True)
            dice1 = tf.reduce_mean(dice1)
            dice1 = tf.maximum(dice1, 1 - dice1)
            dice1, Y_enc_l = permuted_dice(Y_enc_l, mask[0], ignore_black=False)
            dices.append(max(dice1, dice))
            dices2.append(dice if indoor else dice1)
            visualizer.visualize([image[0], mask[0], Y_enc_l])


        stat = report.error_statistics(dices)
        stat2 = report.error_statistics(dices2)
        stats1.append(stat)
        stats2.append(stat2)
        print(correct / cnt)
        print()



    for stat in stats1:
        report.print_report(stat)
    print()
    for stat in stats2:
        report.print_report(stat)
