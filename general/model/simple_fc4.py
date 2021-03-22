import tensorflow as tf
import tensorflow_addons as tfa
from general.model.caffe_alexnet import AlexNet
from general.model.learnable_histogram import DiffHist

class ResizeLayer(tf.keras.layers.Layer):

    def __init__(self, size):
        super().__init__()
        self.size=size

    def call(self, inputs, **kwargs):
        rez = tf.image.resize(inputs, self.size)
        return rez


    def compute_output_shape(self, input_shape):
        return input_shape[0]


    def get_config(self):
        conf = {"size":self.size}
        return conf

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class WeightedAverage(tf.keras.layers.Layer):
    def __init__(self,name, corrected=True):
        super().__init__(name=name)
        self.corrected = corrected

    def call(self, inputs, **kwargs):
        weights = inputs[..., :1]
        values = inputs[...,1:]
        if self.corrected:
            values = tf.nn.l2_normalize(values, -1)
        # values = tf.concat([values_1,values_2,values_3],axis=3)
        weights_shape = tf.shape(weights)
        weights_softmax_shape = (weights_shape[0],tf.reduce_prod(weights_shape[1:]))

        weights = tf.reshape(weights, weights_softmax_shape)
        weights = tf.nn.softmax(weights)
        weights = tf.reshape(weights, weights_shape)

        rez = tf.nn.l2_normalize(tf.reduce_sum(weights*values,axis=[1,2]),axis=-1)

        return rez


    def compute_output_shape(self, input_shape):
        return input_shape[0]


    def get_config(self):
        conf = {"name":self.name}
        return conf


class NewLoss(tf.keras.losses.Loss):

    def __init__(self, name="new_loss", **kwargs):
        super(NewLoss, self).__init__(name=name, **kwargs)


    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def get_config(self):
        return {'name': self.name}

    @tf.function
    def call(self, y_true, y_pred):
        divide = y_pred/y_true
        bias = y_true/y_true
        minus = divide - bias
        _, rez = tf.linalg.normalize(minus,ord=2,axis=1)
        return tf.reduce_sum(rez)


def angle(y_true, y_pred):
    similarity = - tf.keras.losses.cosine_similarity(y_true, y_pred)
    cos = tf.math.acos(similarity)
    angle = cos*180.0/3.14159265359
    return angle



class SimpleFc4(tf.keras.models.Model):

    def __init__(self, size, out_channels=3, weights_path=None, histogram=False, input_mask_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_mask_size = input_mask_size
        self.model = self.build_model(size, out_channels, histogram=histogram)
        if weights_path != None:
            self.model.load_weights(weights_path)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training, mask=mask)

    def build_model(self, size, out_channels, histogram):
        inpts = tf.keras.layers.Input(size, name="in0")

        if self.input_mask_size:
            mask = inpts[..., -self.input_mask_size:]
            inputs = inpts[..., :-self.input_mask_size]
        else:
            inputs = inpts

        layer = inputs

        layer = ResizeLayer((480, 480))(layer)
        layer = tf.keras.layers.Conv2D(240, (1,1), padding="valid", name='branch1_conv')(layer)
        branch_1 = tf.keras.layers.MaxPooling2D(pool_size=(8,8), name='branch1_pool')(layer)

        layer = inputs

        layer = ResizeLayer((492,492))(layer)
        layer = tf.keras.layers.Conv2D(64,(8,8),strides=4,padding='valid', name='branch2_conv1')(layer)
        layer = tf.keras.layers.ReLU()(layer)
        layer = tf.keras.layers.Conv2D(256,(4,4),strides=2,padding='valid', name='branch1_conv2')(layer)
        branch_2 = tf.keras.layers.ReLU()(layer)
        
        if histogram:
            layer = inputs
            layer = ResizeLayer((240, 240))(layer)
            layer = DiffHist(64, name='hist')(layer)
            layer = tf.reshape(layer, (-1, 240, 240, 3*64))
            branch_3 = tf.keras.layers.AveragePooling2D(pool_size=(4, 4), name='hist_pool')(layer)
            branch_3 = tf.keras.layers.BatchNormalization()(branch_3)
        if histogram:
            layer = tf.keras.layers.concatenate([branch_1, branch_2, branch_3])
        else:
            layer = tf.keras.layers.concatenate([branch_1, branch_2])

        layer = tf.keras.layers.Conv2D( filters=256, kernel_size=(5,5), strides=1,
                                        padding='valid', activation='relu',
                                        kernel_initializer='GlorotNormal', name='concat_conv')(layer)


        layer = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(layer)


        layer = tf.keras.layers.Conv2D( filters=64, kernel_size=(6,6), strides=1,
                                        padding='valid', activation='relu',
                                        kernel_initializer='GlorotNormal', name='concat_conv2')(layer)


        layer = tf.keras.layers.Dropout(0.5)(layer)

        layer = tf.keras.layers.Conv2D(filters=out_channels+1, kernel_size=(1,1), strides=1,
                                        padding='valid', activation='relu',
                                        kernel_initializer='GlorotNormal', name='attention_conv')(layer)

        if self.input_mask_size:
            conf_mask = layer[..., -1:]
            pred = layer[..., :-1]
            mask = tf.image.resize(mask, conf_mask.shape[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            new_conf_mask = conf_mask * mask
            layer = tf.concat([pred, new_conf_mask], axis=-1)


        layer = WeightedAverage("0", corrected=True)(layer)

        model = tf.keras.models.Model(inpts, layer)
        return model


class AlexNetFC4(tf.keras.models.Model):

    def __init__(self, size, out_channels=3, weights_path=None, gamma=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.build_model(size, out_channels)
        if weights_path != None:
            self.model.load_weights(weights_path)

        self.scale = 4 if gamma else 1

    def call(self, inputs, training=None, mask=None):
        alex_images = inputs / self.scale

        return self.model(alex_images, training=training, mask=mask)

    def build_model(self, size, out_channels):
        inputs = tf.keras.layers.Input(size, name="in0")
        layer = inputs

        layer = ResizeLayer((512, 1024))(layer)
        an = AlexNet(input_shape=(512, 1024, 3), depth=5, padding='valid', include_top=False, classes=3, use_bn=False)
        an.load_caffe_weights()
        layer = an(layer)

        layer = tf.keras.layers.Conv2D( filters=64, kernel_size=(6,6), strides=1,
                                        padding='valid',
                                        kernel_initializer='GlorotNormal')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)


        layer = tf.keras.layers.Dropout(0.5)(layer)

        layer = tf.keras.layers.Conv2D(filters=out_channels+1, kernel_size=(1,1), strides=1,
                                        padding='valid', activation='relu',
                                        kernel_initializer='GlorotNormal')(layer)

        layer = WeightedAverage("0", corrected=True)(layer)

        model = tf.keras.models.Model(inputs, layer)
        return model

class AlexNetFC4Clustering(tf.keras.models.Model):

    def __init__(self, size, out_channels=3, weights_path=None, gamma=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.build_model(size, out_channels)
        if weights_path != None:
            self.model.load_weights(weights_path)

        self.scale = 4 if gamma else 1

    def call(self, inputs, training=None, mask=None):
        alex_images = inputs / self.scale

        return self.model(alex_images, training=training, mask=mask)

    def build_model(self, size, out_channels):
        inputs = tf.keras.layers.Input(size, name="in0")
        layer = inputs

        layer = ResizeLayer((512, 1024))(layer)
        an = AlexNet(input_shape=(512, 1024, 3), depth=5, padding='valid', include_top=False, classes=3, use_bn=False)
        an.load_caffe_weights()
        layer = an(layer)

        layer = tf.keras.layers.Conv2D( filters=64, kernel_size=(6,6), strides=1,
                                        padding='valid',
                                        kernel_initializer='GlorotNormal')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)


        layer = tf.keras.layers.Dropout(0.5)(layer)

        layer = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(1,1), strides=1,
                                        padding='valid', activation='relu',
                                        kernel_initializer='GlorotNormal')(layer)
        layer = layer / (tf.linalg.norm(layer, 2, -1, True) + 1e-7)
        layer = ResizeLayer(size[0:2])(layer)

        model = tf.keras.models.Model(inputs, layer)
        return model


if __name__ == '__main__':
    an = AlexNetFC4Clustering((256, 512, 3)).model
    an.compile()
    an.summary()
# opt = tfa.optimizers.AdamW(1e-6, learning_rate=0.001, beta_1=0.95,beta_2=0.999)
# model.compile(loss=NewLoss(), optimizer=opt,
#                 metrics=[tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.CosineSimilarity(),angle])
