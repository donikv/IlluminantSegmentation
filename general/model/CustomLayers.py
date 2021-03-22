import tensorflow.keras.layers as l
from general.model.branch_module import *


class FC4(tf.keras.models.Model):
    def __init__(self, out_filters, filters, strides, activation1 ='relu', activation2 ='relu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = l.Conv2D(filters, strides, activation=activation1)
        self.conv2 = l.Conv2D(out_filters + 1, (1, 1), kernel_initializer='random_uniform', activation=activation2)

    def call(self, inputs, training=None, mask=None):
        conv6 = self.conv1(inputs, training=None)
        conv7 = self.conv2(conv6, training=None)
        conf = tf.expand_dims(conv7[..., 0], axis=-1)
        est = conv7[..., 1:]
        weighted = est * conf
        weighted = weighted / (tf.norm(weighted, axis=-1, keepdims=True) + 1e-7)
        return weighted


class ChannelSlice(l.Layer):

    def __init__(self, begin, size=None, **kwargs):
        super().__init__(**kwargs)
        self.begin = begin
        self.size = size

    def call(self, inputs, **kwargs):
        return inputs[..., self.begin] if self.size == None else inputs[..., self.begin:self.begin + self.size]

class IndexLayer(l.Layer):

    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.i = index

    def call(self, inputs, **kwargs):
        return inputs[self.i]

class Slice(l.Layer):

    def __init__(self, begin, size, squeeze_axis=None, **kwargs):
        super().__init__(**kwargs)
        self.begin = begin
        self.size = size
        self.sa = squeeze_axis

    def call(self, inputs, **kwargs):
        sliced = tf.slice(inputs, self.begin, self.size)
        if self.sa is None:
            return sliced
        sliced = tf.squeeze(sliced, axis=self.sa)
        return sliced

class Reshape(l.Layer):

    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape

    def call(self, inputs, **kwargs):
        return tf.reshape(inputs, self.shape)

class Concatenate(l.Layer):

    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.keras.layers.concatenate(inputs, self.axis)


class Tuple(l.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return (*inputs,)

class Tile(l.Layer):

    def __init__(self, multiples, **kwargs):
        super().__init__(**kwargs)
        self.multiples = multiples

    def call(self, inputs, **kwargs):
        return tf.tile(inputs, self.multiples)

class Sum(l.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return sum(inputs)

class Sub(l.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        p = inputs[0]
        for input in inputs[1:]:
            p -= input
        return p

class MulConstant(l.Layer):

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def call(self, inputs, **kwargs):
        return inputs * self.value

class Invert(l.Layer):

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def call(self, inputs, **kwargs):
        return tf.ones_like(inputs) * self.value - inputs


class Mul(l.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        p = inputs[0]
        for input in inputs[1:]:
            p *= input
        return p

class Div(l.Layer):

    def __init__(self, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.e = epsilon

    def call(self, inputs, **kwargs):
        p = inputs[0]
        for input in inputs[1:]:
            p /= (input + self.e)
        return p

class NormLayer(l.Layer):

    def __init__(self, ord=2, **kwargs):
        super().__init__(**kwargs)
        self.ord = ord

    def call(self, inputs, **kwargs):
        p = inputs / (tf.linalg.norm(inputs, axis=-1, ord=self.ord, keepdims=True) + 1e-7)
        return p

class BrightnessNormLayer(l.Layer):

    def __init__(self, ord=2, **kwargs):
        super().__init__(**kwargs)
        self.ord = ord

    def call(self, inputs, **kwargs):
        img_norm = tf.linalg.norm(tf.linalg.norm(tf.linalg.norm(inputs, axis=-1, ord=self.ord, keepdims=True), axis=-2, ord=self.ord, keepdims=True), axis=-3, ord=self.ord, keepdims=True) / 10
        img = inputs / (img_norm + 1e-7)
        return img

class MaskingLayer(l.Layer):

    def __init__(self, invert_mask=False, rb=False, hard_mask=False, **kwargs):
        super().__init__(**kwargs)
        self.invert_mask = invert_mask
        self.rb=rb
        self.hm = hard_mask

    def call(self, inputs, **kwargs):
        inpt = inputs[0]
        mask = inputs[1] if not self.invert_mask else tf.ones_like(inputs[1]) - inputs[1]
        # if self.rb:
        #     masked = inpt[]
        #     return tf.concat(masked)
        if self.hm:
            return tf.where(mask>0.5, inpt*mask, tf.zeros_like(inpt))
        return inpt * mask


class MultiMaskingLayer(l.Layer):

    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, inputs, **kwargs):
        index = self.index
        inpt = inputs[0]
        mask = inputs[1][..., index:index+1]
        return inpt * mask


class ColorCorrectionLayer(l.Layer):

    def __init__(self, normalize=True, rb=False, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize
        self.rb = rb

    def call(self, inputs, **kwargs):
        inpt = inputs[0]
        gt = inputs[1]
        if self.normalize and not self.rb:
            gt = gt / (tf.reduce_max(gt, axis=-1, keepdims=True) + 1e-10)
        gt = tf.expand_dims(gt, axis=1)
        gt = tf.expand_dims(gt, axis=1)
        gt = tf.tile(gt, (1, *inpt.shape[1:3], 1))
        if self.rb:
            gt = tf.stack([gt[...,0], tf.ones_like(gt[...,0]), gt[...,1]], axis=-1)
            corrected = inpt / gt
            # corrected = inpt[..., :gt.shape[-1]] / gt
            # corrected = tf.concat([corrected, inpt[..., gt.shape[-1]:]], axis=-1)
        else:
            corrected = inpt / gt

        return corrected

class CosineSegmentation(l.Layer):

    def __init__(self, normalize=True, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize

    def call(self, inputs, **kwargs):
        inpt = inputs[0]
        gt1 = inputs[1]
        gt2 = inputs[2]

        img = gt1 * (1 - inpt) + gt2 * inpt
        return img

class RBFSimilarity(l.Layer):

    def __init__(self, gamma=1, activation='sigmoid', **kwargs):
        super().__init__(**kwargs)
        self.gamma=gamma
        self.activation = l.Activation(activation=activation)

    def call(self, inputs, **kwargs):
        inpt = inputs[0]
        gt1 = inputs[1]
        gt2 = inputs[2]
        gamma = self.gamma

        d1 = gamma * tf.norm(inpt - gt1, axis=-1, keepdims=True)
        d2 = gamma * tf.norm(inpt - gt2, axis=-1, keepdims=True)

        d1 = 1 / (1 + d1**2)
        d2 = 1 / (1 + d2**2)
        #
        # d1 = tf.exp(-d1**2)
        # d2 = tf.exp(-d2**2)

        return (d2 - d1 + 1) / 2

class GrayWorldEstimationLayer(l.Layer):

    def __init__(self, normalize=True, rb=False, **kwargs):
        super(GrayWorldEstimationLayer, self).__init__(**kwargs)
        self.normalize = normalize
        self.rb = rb

    def call(self, inputs, **kwargs):
        ill = tf.reduce_mean(inputs, axis=[-3,-2])

        if self.normalize and not self.rb:
            ill = ill / (tf.reduce_max(ill, axis=-1, keepdims=True) + 1e-10)

        return ill


class WhitePatchEstimationLayer(l.Layer):

    def __init__(self, normalize=True, rb=False, **kwargs):
        super(WhitePatchEstimationLayer, self).__init__(**kwargs)
        self.normalize = normalize
        self.rb = rb

    def call(self, inputs, **kwargs):
        ill = tf.reduce_max(inputs, axis=[-3,-2])

        if self.normalize and not self.rb:
            ill = ill / (tf.reduce_max(ill, axis=-1, keepdims=True) + 1e-10)

        return ill

class NormConv2D(tf.keras.models.Model):

    def __init__(self, norm='batch', activation='relu', **kwargs):
        name = kwargs['name']
        del kwargs['name']
        super().__init__(name=name)
        self.activation = l.Activation(activation=activation)
        self.norm = l.LayerNormalization() if norm != 'batch' else l.BatchNormalization()
        self.conv = l.Conv2D(**kwargs)

    def build(self, input_shape):
        self.conv.build(input_shape=input_shape)
        conv_out_shape = (*input_shape[:-1], self.conv.filters)
        self.norm.build(input_shape=conv_out_shape)
        self.activation.build(input_shape=conv_out_shape)

    def call(self, inputs, **kwargs):
        y = self.conv(inputs, **kwargs)
        y = self.norm(y, **kwargs)
        y = self.activation(y, **kwargs)
        return y

class Unet(tf.keras.models.Model):

    def __init__(self, **kwargs):
        import segmentation_models as sm
        sm.set_framework('tf.keras')
        super().__init__(name=kwargs['name'])
        del kwargs['name']
        self.unet = sm.Unet(**kwargs)

    def call(self, inputs, training=None, mask=None):
        return self.unet(inputs)

class FPN(tf.keras.models.Model):

    def __init__(self, **kwargs):
        import segmentation_models as sm
        sm.set_framework('tf.keras')
        super().__init__(name=kwargs['name'])
        del kwargs['name']
        self.unet = sm.FPN(**kwargs)

    def call(self, inputs, training=None, mask=None):
        return self.unet(inputs)

class PSP(tf.keras.models.Model):

    def __init__(self, **kwargs):
        import segmentation_models as sm
        sm.set_framework('tf.keras')
        super().__init__(name=kwargs['name'])
        del kwargs['name']
        self.unet = sm.PSPNet(**kwargs)

    def call(self, inputs, training=None, mask=None):
        return self.unet(inputs)

class SharedWeightsLayer(tf.keras.models.Model):

    def __init__(self, original_layer, **kwargs):
        super(SharedWeightsLayer, self).__init__(**kwargs)
        self.og_layer = original_layer

    def call(self, inputs, training=None, mask=None):
        return self.og_layer(inputs)