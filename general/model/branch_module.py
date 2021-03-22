from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

class BranchCell(keras.Model):

    def __init__(self, output, pooling, *args, **kwargs):
        super(BranchCell, self).__init__(*args, **kwargs)
        self.out_size = output
        self.pooling_size = pooling

        self.branch1 = layers.Conv2D(int(output / 2), 3, padding='same', activation='relu')
        self.branch2 = layers.Conv2D(int(output / 2), 2, padding='same', activation='relu')
        self.branch3 = layers.Conv2D(int(output / 2), 1, padding='same', activation='relu')

        self.pool = layers.MaxPooling2D((pooling, pooling), strides=(pooling, pooling))
        self.out = layers.Conv2D(output, 1, strides=(1,1), activation='relu')

    def call(self, inputs, training=None, mask=None):
        b1 = self.branch1(inputs, training=training)
        b2 = self.branch2(inputs, training=training)
        b3 = self.branch3(inputs, training=training)

        b = b1 + b2 + b3
        p = self.pool(b, training=training)
        out = self.out(p, training=training)

        return out


class PyConv(keras.Model):

    def __init__(self, input_features, out_features=[16,16,16,16], input_features_shrink=64, kernel_sizes=[3,5,7,9], groups=[1,4,8,16], activation='sigmoid', regularizer=None, **kwargs):
        super(PyConv, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(input_features_shrink, 1, 1)
        self.bn1 = layers.BatchNormalization()
        self.actv1 = layers.ReLU()
        self.pyconvs = []
        for of, ks, g in zip(out_features, kernel_sizes, groups):
            lvl = layers.Conv2D(of,ks,1, groups=g, padding="same", kernel_regularizer=regularizer, activation=activation, kernel_initializer='zeros')
            self.pyconvs.append(lvl)
        self.bn2 = layers.BatchNormalization()
        self.actv2 = layers.ReLU()
        self.conv2 = layers.Conv2D(input_features, 1, 1)
        self.bn3 = layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        out = self.conv1(inputs)
        out = self.actv1(self.bn1(out))
        pyout = list(map(lambda x: x(out), self.pyconvs))
        pyout = layers.concatenate(pyout, axis=-1, name=self.name + '_concat')
        out = self.actv2(self.bn2(pyout))
        out = self.conv2(out)
        out = self.bn3(out)
        out = out + inputs
        return out


class ResBranch(keras.Model):

    def __init__(self, cardinality, input_channels):
        super(ResBranch, self).__init__()
        self.cardinality = cardinality
        self.branches = []
        for _ in range(cardinality):
            b = keras.Sequential([
                layers.Conv2D(4, 1),
                layers.Conv2D(4, 3, padding='same'),
                layers.Conv2D(input_channels, 1),
                layers.BatchNormalization()
            ])

            self.branches.append(b)

    def call(self, inputs, training=None, mask=None):
        x = sum(list(map(lambda b: b(inputs, training=training), self.branches)))

        x = x + inputs

        return x

def branch_cell(input, output, pooling):
    conv_input = input

    branch11 = layers.Conv2D(int(output / 2), 3, padding='same', activation='relu')(conv_input)
    branch12 = layers.Conv2D(int(output / 2), 2, padding='same', activation='relu')(conv_input)
    branch13 = layers.Conv2D(int(output / 2), 1, padding='same', activation='relu')(conv_input)
    branch1_conc = branch11 + branch12 + branch13 #+ input

    branch1 = layers.MaxPooling2D((pooling, pooling), strides=(pooling, pooling))(branch1_conc)
    branch1 = layers.Conv2D(output, 1, strides=(1,1), activation='relu')(branch1)

    return branch1