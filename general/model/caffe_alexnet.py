import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

layerNames = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
alexnet_weights_path = os.path.join(os.path.dirname(__file__), 'data/bvlc_alexnet.npy')


def conv(input, kernel, biases, c_o, s_h, s_w, padding="VALID", group=1):
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)

    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

class AlexNetConvLayer(tf.keras.models.Model):

    def __init__(self, filters, kernel_size, strides, groups, padding='valid', **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel_size
        self.filters = filters
        self.stride = strides
        self.groups = groups
        self.bias = tf.Variable(tf.zeros((filters,)), name='bias')

        self.convs = []
        for i in range(self.groups):
            self.convs.append(tf.keras.layers.Conv2D(self.filters // self.groups, self.kernel, self.stride, use_bias=False, padding=padding, name=f'cgroup_{i}'))

    def set_weights(self, weights):
        self.bias = tf.Variable(weights[0])
        weights = np.split(weights[1], self.groups, axis=-1)
        for conv, weight in zip(self.convs, weights):
            conv.set_weights([weight])

    def call(self, inputs, **kwargs):
        if self.groups == 1:
            out = self.convs[0](inputs)
        else:
            input_groups = tf.split(inputs, self.groups, axis=-1)
            out_groups = [k(i) for i,k in zip(input_groups, self.convs)]
            out = tf.concat(out_groups, axis=-1)

        out = tf.reshape(tf.nn.bias_add(out, self.bias), [-1] + out.get_shape().as_list()[1:])
        return out

class AlexNetDense(tf.keras.models.Model):

    def __init__(self, filters, kernel_size, strides, groups, padding='valid', **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel_size
        self.filters = filters
        self.stride = strides
        self.groups = groups
        self.bias = tf.Variable(tf.zeros((filters,)), name='bias')

        self.convs = []
        for i in range(self.groups):
            self.convs.append(tf.keras.layers.Conv2D(self.filters // self.groups, self.kernel, self.stride, use_bias=False, padding=padding, name=f'cgroup_{i}'))

    def set_weights(self, weights):
        self.bias = tf.Variable(weights[0])
        weights = np.split(weights[1], self.groups, axis=-1)
        for conv, weight in zip(self.convs, weights):
            conv.set_weights([weight])

    def call(self, inputs, **kwargs):
        if self.groups == 1:
            out = self.convs[0](inputs)
        else:
            input_groups = tf.split(inputs, self.groups, axis=-1)
            out_groups = [k(i) for i,k in zip(input_groups, self.convs)]
            out = tf.concat(out_groups, axis=-1)

        out = tf.reshape(tf.nn.bias_add(out, self.bias), [-1] + out.get_shape().as_list()[1:])
        return out

class LocalResponseNormalization(keras.layers.Layer):

    def __init__(self, radius=2, alpha=2e-05, beta=0.75, bias=1.0, **kwargs):
        super().__init__(**kwargs)
        self.c = lambda x: tf.nn.local_response_normalization(x,
                                          depth_radius=radius,
                                          alpha=alpha,
                                          beta=beta,
                                          bias=bias)

    def call(self, inputs, **kwargs):
        return self.c(inputs)


class AlexNet(keras.Model):


    def __init__(self, input_shape=(227, 227, 3), padding='valid', depth=8, include_top=True, classes=102, use_bn=False):
        super().__init__()
        self.depth = depth
        norm = keras.layers.BatchNormalization if use_bn else LocalResponseNormalization

        model = keras.Sequential()
        model.add(keras.layers.Input(input_shape))
        # First layer: Convolutional layer with max pooling and batch normalization.
        model.add(AlexNetConvLayer(kernel_size=(11, 11),
                                      strides=(4, 4),
                                      padding=padding,
                                      filters=96,
                                      name=layerNames[0], groups=1))
        model.add(keras.layers.ReLU())
        model.add(norm())
        model.add(keras.layers.MaxPooling2D(pool_size=(3, 3),
                                            strides=(2, 2),
                                            padding="valid"))
        if depth > 1:
            # Second layer: Convolutional layer with max pooling and batch normalization.
            model.add(AlexNetConvLayer(kernel_size=(5, 5),
                                          strides=(1, 1),
                                          padding=padding,
                                          filters=256,
                                          name=layerNames[1], groups=2))
            model.add(keras.layers.ReLU())
            model.add(norm())
            model.add(keras.layers.MaxPooling2D(pool_size=(3, 3),
                                                strides=(2, 2),
                                                padding="valid"))
        if depth > 2:
            # Third layer: Convolutional layer with batch normalization.
            model.add(AlexNetConvLayer(kernel_size=(3, 3),
                                          strides=(1, 1),
                                          padding=padding,
                                          filters=384, name=layerNames[2], groups=1))
            model.add(keras.layers.ReLU())

        if depth > 3:
            # Fourth layer: Convolutional layer with batch normalization.
            model.add(AlexNetConvLayer(kernel_size=(3, 3),
                                          strides=(1, 1),
                                          padding=padding,
                                          filters=384, name=layerNames[3], groups=2))
            model.add(keras.layers.ReLU())

        if depth > 4:
            # Fifth layer: Convolutional layer with max pooling and batch normalization.
            model.add(AlexNetConvLayer(kernel_size=(3, 3),
                                          strides=(1, 1),
                                          padding=padding,
                                          filters=256, name=layerNames[4], groups=2))
            model.add(keras.layers.ReLU())
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                strides=(2, 2),
                                                padding="valid"))

        if depth > 5:
            # Flatten the output to feed it to dense layers
            model.add(keras.layers.Flatten())

            # Sixth layer: fully connected layer with 4096 neurons with 40% dropout and batch normalization.
            model.add(keras.layers.Dense(units=4096,
                                         activation=tf.nn.relu, name=layerNames[5]))
            model.add(keras.layers.Dropout(rate=0.4))
            model.add(keras.layers.BatchNormalization())

        if depth > 6:
            # Seventh layer: fully connected layer with 4096 neurons with 40% dropout and batch normalization.
            model.add(keras.layers.Dense(units=4096,
                                         activation=tf.nn.relu, name=layerNames[6]))
            model.add(keras.layers.Dropout(rate=0.4))
            model.add(keras.layers.BatchNormalization())

        if depth > 7:
            # Eigth layer: fully connected layer with 1000 neurons with 40% dropout and batch normalization.
            model.add(keras.layers.Dense(units=1000,
                                         activation=tf.nn.relu, name=layerNames[7]))
            model.add(keras.layers.Dropout(rate=0.4))
            model.add(keras.layers.BatchNormalization())
            if include_top:
                # Output layer: softmax function of 102 classes of the dataset. This integer should be changed to match
                # the number of classes in your dataset if you change from Oxford_Flowers.
                model.add(keras.layers.Dense(units=classes,
                                             activation=tf.nn.softmax))

        self.model = model

    def call(self, inputs, **kwargs):
        alex_images = (inputs * 4.0 *
                       255.0)[:, :, :, ::-1]
        return self.model(alex_images)

    def load_caffe_weights(self, path=alexnet_weights_path):
        weights_dic = np.load(path, encoding='bytes', allow_pickle=True).item()
        for i in range(min(self.depth, 8)):
            w = weights_dic[layerNames[i]][0]
            b = weights_dic[layerNames[i]][1]

            layer = list(filter(lambda l: l.name == layerNames[i], self.model.layers))[0]
            layer.set_weights([b, w])


if __name__ == '__main__':
    import numpy as np
    weights_dic = np.load('data/bvlc_alexnet.npy', encoding='bytes', allow_pickle=True).item()
    a = AlexNet((227, 227, 3), depth=4)
    a.load_caffe_weights('data/bvlc_alexnet.npy')