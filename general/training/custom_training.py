import tensorflow as tf
from general.datasets import TauDataset, Cube2Dataset, CubeDataset, NewCube2Dataset, CubeNDataset
from configparser import ConfigParser, ExtendedInterpolation
from general.processing import data_processing as dp
import numpy as np
import os
from sklearn.model_selection import train_test_split
from general.training import metrics, optimizers, losses
from copy import deepcopy
from general.model.CustomModel import try_parse, try_parse_or_default, build_model_from_config
from general.utils.configs import dotdict
import inspect


def get_dataset(config,type_, paths_filter=lambda x: x, out_mapping=lambda *x: x):
    dt = config['dataset_type'].lower().strip()
    dataset_path = config['dataset_path'].strip()

    batch_size = try_parse_or_default(config, 'batch_size', 1)
    size = try_parse(config['size'])

    if dt == "cube2":
        regression = try_parse_or_default(config, 'regression', False)
        return_gt = try_parse_or_default(config, 'return_gt', True)
        return_gt_mask = try_parse_or_default(config, 'return_gt_mask', False)
        round_mask = try_parse_or_default(config, 'round', True)
        gamma = try_parse_or_default(config, 'gamma', False)
        resample = try_parse_or_default(config, 'resample', 1)
        camera = try_parse_or_default(config, 'camera', None)
        scene_type = try_parse_or_default(config, 'scene_type', None)

        im_paths = paths_filter(Cube2Dataset.load_paths(dataset_path, config['list_name']))
        ds = Cube2Dataset.dataset(im_paths, type_, batch_size, cache=False,
                                  regression=regression, gt=return_gt, gt_mask=return_gt_mask,
                                  round=round_mask, gamma=gamma, resample=resample,
                                  camera=camera, scene_type=scene_type,
                                  h=size[0], w=size[1], map_fn=out_mapping)
        if type(ds) == tuple:
            ds, im_paths = ds
    elif dt == "newcube2":
        regression = try_parse_or_default(config, 'regression', False)
        return_gt = try_parse_or_default(config, 'return_gt', True)
        return_gt_mask = try_parse_or_default(config, 'return_gt_mask', False)
        round_mask = try_parse_or_default(config, 'round', True)
        gamma = try_parse_or_default(config, 'gamma', False)
        resample = try_parse_or_default(config, 'resample', 1)
        camera = try_parse_or_default(config, 'camera', None)
        scene_type = try_parse_or_default(config, 'scene_type', None)

        im_paths = paths_filter(Cube2Dataset.load_paths(dataset_path, config['list_name']))
        ds = NewCube2Dataset.dataset(im_paths, type_, batch_size, cache=False,
                                     regression=regression, gt=return_gt, gt_mask=return_gt_mask,
                                     round=round_mask, gamma=gamma, resample=resample,
                                     camera=camera, scene_type=scene_type,
                                     h=size[0], w=size[1], map_fn=out_mapping)
        if type(ds) == tuple:
            ds, im_paths = ds
    elif dt == "cuben":
        regression = try_parse_or_default(config, 'regression', False)
        return_gt = try_parse_or_default(config, 'return_gt', True)
        round_mask = try_parse_or_default(config, 'round', True)
        gamma = try_parse_or_default(config, 'gamma', False)
        n = try_parse_or_default(config, 'n', 3)

        im_paths = paths_filter(Cube2Dataset.load_paths(dataset_path, config['list_name']))
        ds = CubeNDataset.dataset(im_paths, type_, batch_size, cache=False,
                                  regression=regression, gt=return_gt, round=round_mask, gamma=gamma,
                                  h=size[0], w=size[1], n=n, map_fn=out_mapping)
    elif dt == 'cube':
        im_paths = paths_filter(CubeDataset.load_image_names(config['list_name'], dataset_path))
        gts = np.loadtxt(os.path.join(dataset_path, config['gts_name']))
        indices = np.array(list(map(lambda x: int(x[x.rfind('/') + 1:-4]) - 1, im_paths)))

        ds = CubeDataset.regression_dataset(im_paths, indices, type=type_,
                                            bs=batch_size, cache=False, uv=False,
                                            gts=gts, map_fn=out_mapping, sz=size)
    elif dt == 'tau':
        im_paths = np.loadtxt(os.path.join(dataset_path, config['list_name']), dtype=str)
        im_paths = paths_filter(im_paths)
        im_paths = list(map(lambda x: os.path.join(dataset_path, x[1:]), im_paths))

        ds = TauDataset.regression_dataset(im_paths, type=type_, bs=batch_size,
                                           cache=False, uv=False,
                                           sz=size, map_fn=out_mapping)

    else:
        raise ValueError(f"Unknown dataset type: {dt}")
    return ds, len(im_paths)


def get_dataset_split(config):
    ft = config['split'].lower().strip()
    testf = try_parse(config['test'])
    trainf = try_parse(config['train'])
    validf = try_parse(config['valid'])

    if ft == 'train_test_split':
        seed = try_parse_or_default(config, 'seed', 42)
        def split(images):
            tv, test = train_test_split(images, train_size=1-testf, test_size=testf, random_state=seed)
            train, valid = train_test_split(tv, train_size=trainf, test_size=validf, random_state=seed)
            return {dp.TRAIN:train, dp.VALID:valid, dp.TEST:test}
    elif ft == 'filter':
        def split(images):
            train = np.array(list(filter(lambda x: x.find(trainf) != -1, images)))
            valid = np.array(list(filter(lambda x: x.find(validf) != -1, images)))
            test = np.array(list(filter(lambda x: x.find(testf) != -1, images)))
            return {dp.TRAIN:train, dp.VALID:valid, dp.TEST:test}
    else:
        raise ValueError(f"Unknown filter type: {ft}")
    return split

def get_dataset_mapping(config):
    config = deepcopy(config)
    mt = config['mapping'].lower().strip()
    del config['mapping']
    props = {k: try_parse(v) for k, v in config.items()}

    if mt == 'image_histogram_mapping_segmentation':
        f = lambda *x: dp.image_histogram_mapping_segmentation(*x, **props)
    elif mt == 'image_histogram_mapping':
        f = lambda *x: dp.image_histogram_mapping(*x, **props)
    else:
        try:
            func = getattr(dp, mt)
            f = lambda *x: func(*x, **props)
        except:
            raise ValueError(f"Unknown dataset mapping: {mt}")
    return f

def get_input_mapping(config):
    config = deepcopy(config)
    mt = config['mapping'].lower().strip()
    del config['mapping']
    props = {k: try_parse(v) for k, v in config.items()}

    try:
        func = getattr(dp, mt)
        f = lambda dataset: func(dataset, **props)
    except:
        raise ValueError(f"Unknown dataset mapping: {mt}")
    return f


def get_loss(config):
    config = deepcopy(config)
    loss_name = config['name'].strip()
    del config['name']

    loss_weight = try_parse_or_default(config, "weight", default=1.0)
    try:
        del config['weight']
    except:
        pass

    props = {k: try_parse(v) for k, v in config.items()}

    try:
        loss_cls = getattr(losses, loss_name)
        loss = loss_cls(**props)
        return loss, loss_weight
    except:
        raise ValueError(f"Unknown loss: {loss_name}")


def get_optimizer(config):
    config = deepcopy(config)
    name = config['name'].strip()
    del config['name']
    props = {k: try_parse(v) for k, v in config.items()}

    try:
        opt_class = getattr(optimizers, name)
        opt = opt_class(**props)
        return opt
    except:
        raise ValueError(f"Unknown optimizer: {name}")

def get_metrics(config):
    def load_metric(name):
        try:
            metric_cls = getattr(metrics, name)
            metric = metric_cls() if inspect.isclass(metric_cls) else metric_cls
            return metric
        except:
            raise ValueError(f"Unknown metric: {name}")

    names = try_parse(config['names'].strip())
    if type(names) is str:
        return [load_metric(names)]
    else:
        return list(map(lambda x: load_metric(x), names))

def get_parameters(config):
    return {k.lower().strip(): try_parse(v) for k, v in config.items()}


class CustomTraining():

    def __init__(self, training_config_path):
        self.training_config = ConfigParser(interpolation=ExtendedInterpolation())
        self.training_config.read(training_config_path)

    def get_dataset_mapping(self):
        try:
            input_mapping = get_input_mapping(self.training_config["input_mapping"])
        except:
            input_mapping = lambda x: x
        return input_mapping

    def get_dataset_from_config(self, type=dp.TRAIN):
        f = lambda x: get_dataset_split(self.training_config['split'])(x)[type]
        mapping = get_dataset_mapping(self.training_config['mapping'])
        out_mapping = lambda *x: mapping(*x)
        ds, length = get_dataset(self.training_config[type], type, f, out_mapping)
        # try:
        ds_map_fn = self.get_dataset_mapping()
        ds = ds_map_fn(ds)
        # except:
        #     pass
        return ds, length

    def get_fit_parameters_from_config(self):
        loss_names = list(filter(lambda x: x.find('loss') != -1, self.training_config.sections()))
        losses = []
        weights = []
        for loss_name in loss_names:
            loss, weight = get_loss(self.training_config[loss_name])
            losses.append(loss)
            weights.append(weight)
        if len(losses) == 1:
            losses = losses[0]
            weights = weights[0]

        opt = get_optimizer(self.training_config['optimizer'])

        metrics_names = list(filter(lambda x: x.find('metric') != -1, self.training_config.sections()))
        mtrcs = []
        for metrics_name in metrics_names:
            metrics = get_metrics(self.training_config[metrics_name])
            mtrcs.append(metrics)
        if len(mtrcs) == 1:
            mtrcs = mtrcs[0]
        return {'loss': losses, 'loss_weights': weights, 'optimizer': opt, 'metrics': mtrcs}

    def load_model_from_config(self):
        folder = self.training_config['model']['folder']
        config_path = f'{folder}/model.config'

        model_config = ConfigParser(interpolation=ExtendedInterpolation())
        model_config.read(config_path)
        size = try_parse_or_default(self.training_config['parameters'], 'size', (256, 512))
        act = try_parse_or_default(self.training_config['model'], 'activation', 'softmax')
        classes = try_parse_or_default(self.training_config['model'], 'classes', 3)
        model_config['config']['H'] = str(size[0])
        model_config['config']['W'] = str(size[1])
        model_config['config']['HW'] = str(size[0] * size[1])
        model_config['config']['ACTIVATION'] = act
        model_config['config']['CLASSES'] = str(classes)

        for key, value in self.training_config['model'].items():
            model_config['config'][key.upper()] = value


        model = build_model_from_config(model_config)
        model.summary()
        tf.keras.utils.plot_model(model, f"{folder}/model.png", show_shapes=True)
        pretrained_path = try_parse_or_default(self.training_config['model'], 'pretrained_path', None)
        if pretrained_path is not None:
            model.load_weights(pretrained_path)
        return model

    def get_training_parameters(self):
        params = get_parameters(self.training_config['parameters'])
        return dotdict(params)

def train(model, train_dataset, loss_fn, epochs, optimizer):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                logits = model(x_batch_train, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(tf.cast(y_batch_train, float), logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * 64))
