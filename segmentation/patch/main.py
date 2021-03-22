from loader import *
from sklearn.preprocessing import *
from general.utils.visualizer import *
from skimage.io import *
from sklearn.model_selection import train_test_split

import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime

if __name__ == '__main__':

    # %%
    
    ap = argparse.ArgumentParser(description='Train rf classifier')

    ap.add_argument("--dataset_path", required=True,
                    help="Path to the dataset folder")

    ap.add_argument("--dataset_list", required=True,
                    help="List of dataset images to use")

    args = vars(ap.parse_args())

    dataset_path = args['dataset_path']
    dataset_list = args['dataset_list']

    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                                  'darkorange'])


    def plot_results(X, Y_, means, covariances, index, title):
        splot = plt.subplot(2, 1, 1 + index)
        for i, (mean, covar, color) in enumerate(zip(
                means, covariances, color_iter)):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)

        plt.xticks(())
        plt.yticks(())
        plt.title(title)


    def plot_clusters(X, Y_, centers, weights, title):
        for i, (mean, color) in enumerate(zip(
                centers, color_iter)):
            if not np.any(Y_ == i):
                continue
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], color=color, s=weights[Y_ == i])

        plt.xticks(())
        plt.yticks(())
        plt.title(title)

    # %%

    paths = load_paths(dataset_path, dataset_list)
    paths, paths_test = train_test_split(paths, train_size=0.8, test_size=0.2, random_state=42)
    # shuffle(paths)
    # paths_test = paths[-30:]
    # paths = paths[:100]

    img_paths = list(map(lambda x: x + '/img.png', paths))
    mask_paths = list(map(lambda x: x + '/gt_mask_round.png', paths))

    images = img_paths
    gts = np.array(list(map(lambda x: np.loadtxt(x + '/gt.txt'), paths)))
    masks = mask_paths
    print("Extracting features...")
    features_com1 = img_to_features(images, gts, n_jobs=8)

    # %%

    num_features = features_com1.shape[-1]
    X = features_com1.reshape((-1, num_features))
    print("Creating targets...")
    y = target_creator(masks)

    enc = OneHotEncoder(sparse=False)
    y_enc = enc.fit_transform(y.reshape((-1, 1)))

    # %%

    visualize(y_enc.reshape((y.shape[0], 30, 32, 3))[0:2])

    # %% md

    # Classification

    # %%

    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    from sklearn.ensemble import *
    from joblib import dump, load
    import multiprocessing
    N_JOBS = multiprocessing.cpu_count() - 1

    # %%

    scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y.reshape((-1,)), test_size=.4, random_state=42)

    # SVC
    cs = np.logspace(-3, 6, num=10, base=10)
    cs = [10, 100, 1000]
    gammas = np.logspace(-1, 5, num=6, base=10)
    gammas = [1.585, 2]
    clf = SVC(probability=True, class_weight='balanced')
    params = {'gamma': gammas, 'C': cs}

    n_est = [100, 200, 500, 1000]
    clf = RandomForestClassifier(n_estimators=500)
    params = {'n_estimators': n_est}

    # ADA BOOST
    # lrs = np.logspace(-3,  2, num=6, base=2)
    # params = {'n_estimators':[25, 50, 75, 85, 100, 110], "learning_rate":lrs}
    # clf = AdaBoostClassifier()
    #
    clf = GridSearchCV(clf, params, verbose=3, n_jobs=N_JOBS, cv=4)
    clf.fit(X_train, y_train)
    clf_b = clf.best_estimator_
    y_pred = clf_b.predict(X_test)
    print(classification_report(y_test, y_pred))

    # GAUSSIAN
    # clf = GaussianNB()
    # scores = cross_validate(clf, X_train, y_train, n_jobs=-1, scoring='f1_macro', return_estimator=True)
    # print(scores)
    # clf_b = clf
    # y_pred = clf_b.predict(X_test)
    # print(classification_report(y_test, y_pred))
    # clfs = []
    # for c in cs:
    #     print(c)
    #     clf = SVC(gamma=2, C=c)#SVC(kernel="linear", C=c)
    #     clf.fit(X_train, y_train)
    #
    #     y_pred = clf.predict(X_test)
    #     clfs.append(clf)
    #
    # sorted(clfs, key=lambda x: x.score)

    # %%

    # paths_test = load_paths('D:/fax/Cube2/','list_indoor.txt')
    # paths_test = paths[-10:]
    # paths = paths[0:50]
    img_paths_test = list(map(lambda x: x + '/img.png', paths_test))
    mask_paths_test = list(map(lambda x: x + '/gt_mask_round.png', paths_test))

    images_test = img_paths_test
    gts_test = np.array(list(map(lambda x: np.loadtxt(x + '/gt.txt'), paths_test)))
    masks_test = mask_paths_test


    features_com_t = img_to_features(images_test, gts_test, n_jobs=8)


    # %%

    X_t = features_com_t.reshape((-1, num_features))
    y_t = target_creator(masks_test)
    y_enc_t = enc.fit_transform(y_t.reshape((-1, 1)))

    # %%

    # X_t_s = StandardScaler().fit_transform(X_t)
    # X_t = scaler.transform(X_t)

    y_t_pred = clf_b.predict(X_t)

    # %%

    print(y_t_pred)
    print(classification_report(y_t.reshape((-1,)), y_t_pred))
    y_t_pred_e = enc.transform(y_t_pred.reshape((-1, 1)))
    masks_pred = y_t_pred_e.reshape((len(masks_test), 30, 32, 3))
    visualize(list(map(lambda x: imread(x), images_test)))
    visualize(masks_pred)

    # %%

    visualize(y_enc_t.reshape((len(masks_test), 30, 32, 3)))

    # %% md

    ## Saving model

    # %%

    os.makedirs('models/classifiers', exist_ok=True)
    dump(clf_b, f'models/classifiers/{datetime.datetime.now().strftime("%Y%m%d-%H%M")}_rf.joblib')
    # dump(scaler, 'models/scalers/rand_forrest_14_12_all_scaled.joblib')

    # %%

