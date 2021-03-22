import tensorflow as tf
from general.training import metrics
from general.processing import histogram
from tensorflow.keras.losses import *

MSE = MeanSquaredError
MAE = MeanAbsoluteError
BCE = BinaryCrossentropy

def balanced_cross_entropy(beta=1):
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        pos_weight = beta / (1 - beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss * (1 - beta))

    return loss


@tf.function
def cosine_similarity(l1, l2, eps=1e-7, keepdims=False):
    l1_norm = tf.norm(l1, axis=-1, keepdims=True) + eps
    l2_norm = tf.norm(l2, axis=-1, keepdims=True) + eps
    l1l2 = tf.reduce_sum(l1*l2, axis=-1, keepdims=True)
    cos = (l1l2) / (l1_norm*l2_norm)
    cos = tf.clip_by_value(cos, -1, 1)
    return tf.squeeze(tf.acos(cos)) if not keepdims else tf.acos(cos)


class DiceLoss(tf.losses.Loss):

    def call(self, y_true, y_pred):
        return metrics.dice_coef_loss(y_true, y_pred)


class BCEDice(tf.losses.Loss):

    def call(self, y_true, y_pred):
        def dice_loss(y_true, y_pred):
            numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
            denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

            return tf.reshape(1 - (numerator + 1) / (denominator + 1), (-1, 1, 1))

        return tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True) + dice_loss(y_true, y_pred) / 4


class CEDice(tf.losses.Loss):

    def __init__(self, a=1, b=1/3):
        super(CEDice, self).__init__()
        self.a = a
        self.b = b

    def call(self, y_true, y_pred):
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred) * self.a + \
               tf.reduce_mean(1 - metrics.dice_coef(y_true, y_pred, num_classes=3)) * self.b


class CECosineLoss(tf.losses.Loss):

    def __init__(self, out_shape, apply_softmax=False, depth=16, center_u=0., center_v=0., eps=histogram.EPS, multi_loss=False, model=None):
        super(CECosineLoss, self).__init__()
        self.apply_softmax = apply_softmax
        self.depth = depth
        self.color_table = histogram.color_table(depth, center_u=center_u, center_v=center_v, eps=eps)

        self.color_table_flat = tf.reshape(self.color_table, (-1, 3))

        self.cos = tf.losses.CosineSimilarity(axis=-1)
        self.cc = tf.losses.CategoricalCrossentropy()
        self.os = tf.constant(out_shape)
        self.eps = eps
        self.multi_loss = multi_loss
        self.model = model


    def call(self, y_true, y_pred):
        # y_pred = tf.sqrt(y_pred)
        e = None
        if self.apply_softmax:
            y_pred = tf.nn.softmax(y_pred)
        y_pred = y_pred / (tf.reduce_sum(y_pred, axis=-1, keepdims=True) + 1e-10)

        tf.debugging.assert_all_finite(
            y_pred, 'Predicted tensor must not contain nan or inf values', name=None
        )

        if self.multi_loss:
            e = self.__multi_loss__(e, y_pred, y_true)
        else:
            e = self.__single_loss__(e, y_pred, y_true)

        tf.debugging.assert_all_finite(
            e, 'Loss must not have nan values', name=None
        )

        l2_reg = 0 if self.model is None else tf.math.add_n(self.model.losses)
        return e + l2_reg

    def __bn__(self, x):
        bins = histogram.bn(x, depth=self.depth)
        bins = tf.squeeze(bins)

        # bins = bins[0] * self.depth + bins[1]
        return tf.cast(bins, tf.float32)

    def __multi_loss__(self, e, y_pred, y_true):
        if y_true.shape[-1] == 6:
            y_true = tf.reshape(y_true, (-1, 2, 3))
            y_true, _ = histogram.to_uv(y_true)
        elif y_true.shape[-1] == 4:
            y_true = tf.reshape(y_true, (-1, 2, 2))

        # ytrue: (bs x 2 x 2)

        y_true = tf.map_fn(lambda x: tf.stack((self.__bn__(x[0]), self.__bn__(x[1]))), y_true)
        # print(y_true.shape)
        index_true = tf.cast(y_true, tf.int32)

        tf.debugging.assert_all_finite(
            y_true, 'Target tensor must not contain nan or inf values', name=None
        )
        rgb_true = tf.gather(self.color_table_flat, index_true)

        cos1 = create_loss_ct(rgb_true[:, 0, :], self.os, self.depth, self.color_table_flat)
        cos2 = create_loss_ct(rgb_true[:, 1, :], self.os, self.depth, self.color_table_flat)
        cos = 2 / (1/cos1 + 1/cos2)

        # l1 = tf.sqrt(cos1) * y_pred
        # l2 = tf.sqrt(cos2) * y_pred
        l = cos * y_pred
        tf.debugging.assert_all_finite(
            l, 'Loss must not have nan values', name=None
        )
        # l = l1 + l2

        if e == None:
            e = tf.reduce_sum(l)
        else:
            e = e + tf.reduce_sum(l)
        return e  # + self.cc(y_true, y_pred)


    def __single_loss__(self, e, y_pred, y_true):
        if y_true.shape[-1] == 3:
            y_true, _ = histogram.to_uv(y_true)

        # ytrue: (bs x 2)

        y_true = tf.map_fn(self.__bn__, y_true)
        # print(y_true.shape)
        index_true = tf.cast(y_true, tf.int32)
        tf.debugging.assert_all_finite(
            y_true, 'Target tensor must not contain nan or inf values', name=None
        )

        rgb_true = tf.gather(self.color_table_flat, index_true)
        if len(rgb_true.shape) == 3 and rgb_true.shape[1] == 1:
            rgb_true = rgb_true[:, 0, :]

        cos = create_loss_ct(rgb_true, self.os, self.depth, self.color_table_flat) #/ 3.14 * 180
        l = cos * y_pred

        if e == None:
            e = tf.reduce_sum(l)
        else:
            e = e + tf.reduce_sum(l)
        return e


def illuminant_loss(y_true, y_pred):
    a = y_pred / (y_true + 1e-10)
    b = tf.norm(a-1, axis=-1)

    return b

class IlluminantLoss(tf.keras.losses.Loss):

    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        if y_true.shape[-1] > 3 and len(y_true.shape) == 2:
            n_ill = y_true.shape[-1] // 3
            y_true = tf.reshape(y_true, (-1, n_ill, 3))
            y_pred = tf.reshape(y_pred, (-1, n_ill, 3))
        return illuminant_loss(y_true, y_pred)

def create_loss_ct(rgb_true, os, depth, color_table_flat=None):
    if color_table_flat is None:
        color_table = histogram.color_table(depth)
        color_table_flat = tf.reshape(color_table, (-1, 3))
    b = tf.concat([tf.ones(len(os) - 1, dtype=tf.int32), tf.constant([depth ** 2])], axis=0)
    rgb_true = tf.tile(rgb_true, b)
    rgb_shape = tf.concat([os, tf.constant([3])], axis=0)
    rgb_true = tf.reshape(rgb_true, rgb_shape)

    b = tf.concat([tf.reduce_prod(os[:-1], keepdims=True), tf.constant([1])], axis=0)
    color_table_flat = tf.tile(color_table_flat, b)
    c_shape = tf.concat([os[:-1], tf.constant([depth * depth, 3])], axis=0)
    color_table_flat = tf.reshape(color_table_flat, c_shape)
    cos = cosine_similarity(rgb_true, color_table_flat)

    return cos

class CosMSELoss(tf.keras.losses.Loss):

    def __init__(self, a=1, b=4, two_ills=False, out_channels=2, cross_data=False, mode='uv'):
        super(CosMSELoss, self).__init__()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.cos = tf.keras.losses.CosineSimilarity()
        self.a = a; self.b = b
        self.two_ills = two_ills
        self.oc = out_channels
        self.x = cross_data
        self.mode = mode

    def call(self, y_true, y_pred):
        if self.two_ills:
            shp = self.oc
            y_true_1, y_true_2 = y_true[..., 0:shp], y_true[..., shp:shp*2]
            y_pred_1, y_pred_2 = y_pred[..., 0:shp], y_pred[..., shp:shp*2]
            if shp != 3:
                y_true_1, y_true_2 = histogram.from_uv(y_true_1), histogram.from_uv(y_true_2)
                y_pred_1, y_pred_2 = histogram.from_uv(y_pred_1), histogram.from_uv(y_pred_2)
            e1 = self.a * self.mse(y_true_1, y_pred_1) + self.b * (cosine_similarity(y_true_1, y_pred_1) * 180 / 3.14)
            e2 = self.a * self.mse(y_true_2, y_pred_2) + self.b * (cosine_similarity(y_true_2, y_pred_2) * 180 / 3.14)
            e = e1 + e2
            if self.x:
                e1 = self.b * 1.0 * (cosine_similarity(y_true_1 - y_true_2, y_pred_1 - y_pred_2) * 180 / 3.14)
                e = e + e1
            return e
        if self.mode == 'uv':
            y_pred, y_true = histogram.from_uv(y_pred), histogram.from_uv(y_true)
        elif self.mode == 'rb':
            y_pred, y_true = histogram.from_rb(y_pred), histogram.from_rb(y_true)
        return self.a * self.mse(y_true, y_pred) + self.b * (cosine_similarity(y_true, y_pred) * 180 / 3.14)


class MaskRegressionLoss(tf.keras.losses.Loss):

    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        y_true = tf.reduce_max(y_true, axis=-1, keepdims=True)
        y_pred = tf.reduce_max(y_pred, axis=-1, keepdims=True)

        return mse(y_true, y_pred)


class MaskClassificationLoss(tf.keras.losses.Loss):

    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        y_true = tf.reduce_max(y_true, axis=-1, keepdims=True)
        y_pred = tf.reduce_max(y_pred, axis=-1, keepdims=True)

        return binary_crossentropy(y_true, y_pred)

class MultiMaskRegressionLoss(tf.keras.losses.Loss):

    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)


class CompositeLoss(tf.losses.Loss):
    def __init__(self, losses, weights):
        super(CompositeLoss, self).__init__()

        self.losses = losses
        self.weights = weights

    def call(self, y_true, y_pred):
        l = 0
        for loss, weight in zip(self.losses, self.weights):
            l += weight * tf.reduce_sum(loss(y_true, y_pred))

        return l


class PretrainingLoss(tf.keras.losses.Loss):

    def __init__(self, seg_dimensions, reg_dimensions):
        super().__init__()
        self.seg = seg_dimensions
        self.reg = reg_dimensions
        self.loss = IlluminantLoss()

    def call(self, y_true, y_pred):
        pred_ill = tf.reduce_mean(y_pred[..., self.seg:], axis=(1,2))

        return self.loss(y_true, pred_ill)




class UNetLoss(tf.keras.losses.Loss):

    def __init__(self, seg_dimensions, reg_dimensions, seg_loss='dice', reg_loss=IlluminantLoss(), reg_head=True, multi_ill=True, ignore_black=False, pretraining=False):
        super().__init__()
        self.output_shapes = seg_dimensions
        self.reg_dim = reg_dimensions
        self.seg_loss = CEDice(a=1, b=1) if seg_loss == 'dice' else MaskRegressionLoss()
        self.reg_loss = reg_loss
        self.reg_head = reg_head
        self.multi_ill = multi_ill
        self.ignore_black = ignore_black
        self.pretraining = pretraining

    def call(self, y_true, y_pred):
        if self.reg_head:
            y_true_seg = y_true[..., :self.output_shapes]
            y_pred_seg = y_pred[..., :self.output_shapes]

            if self.ignore_black:
                y_true_seg = tf.stack([y_true_seg[..., 0], y_true_seg[..., 2]], axis=-1)
                y_pred_seg = tf.stack([y_pred_seg[..., 0], y_pred_seg[..., 2]], axis=-1)

            seg_l = self.seg_loss(y_true_seg, y_pred_seg)

            y_pred_reg = y_pred[..., self.output_shapes:]
            y_pred_reg = tf.reduce_mean(y_pred_reg, axis=[1, 2])
            if self.multi_ill:
                y_true_reg = y_true[..., self.output_shapes:self.output_shapes + self.reg_dim]
                y_true_reg2 = y_true[..., self.output_shapes + self.reg_dim:]
                y_true_reg = tf.reduce_mean(y_true_reg, axis=[1, 2])
                y_true_reg2 = tf.reduce_mean(y_true_reg2, axis=[1, 2])

                reg_l = tf.minimum(self.reg_loss(y_true_reg, y_pred_reg), self.reg_loss(y_true_reg2, y_pred_reg))
            else:
                y_true_reg = y_true[..., self.output_shapes:self.output_shapes + self.reg_dim]
                y_true_reg = tf.reduce_mean(y_true_reg, axis=[1, 2])
                reg_l = self.reg_loss(y_true_reg, y_pred_reg)

            if self.pretraining:
                return tf.reduce_sum(reg_l)
            return tf.reduce_sum(reg_l) + tf.reduce_sum(seg_l)
        else:
            return tf.reduce_sum(self.seg_loss(y_true, y_pred))

@tf.function
def delete_tf(a, idx, axis=0):
    a1 = tf.cast(idx, dtype=tf.bool)
    mask = tf.math.logical_not(a1)
    return tf.boolean_mask(a, mask, axis=axis)

@tf.function
def cluster_illuminant_loss_old(mask, gts, y_pred, n_ill=3, alpha=0.5):
    @tf.function
    def pixel_loss(data):
        idx = tf.cast(data[3:3+n_ill], tf.int32)
        y_pred = data[:3]
        gts = tf.reshape(data[3+n_ill:], (-1,3))
        gts_other = delete_tf(gts, idx)
        t = illuminant_loss(gts[tf.argmax(idx)], y_pred)
        print(t, y_pred, gts, gts_other)
        f = illuminant_loss(gts_other, y_pred)

        loss = tf.maximum(t - 1 / (n_ill - 1) * tf.reduce_sum(f) + alpha, 0)
        return loss

    # mask = tf.argmax(mask, axis=-1)[..., tf.newaxis]
    mask = tf.cast(mask, float)

    data = tf.concat([y_pred, mask, gts],axis=-1)
    data = tf.reshape(data, (-1, data.shape[-1]))
    print(data)

    ls = tf.map_fn(pixel_loss, data)
    return ls


#@tf.function
def cluster_illuminant_loss(mask, gts, y_pred, n_ill=3, alpha=0.5):

    gts = tf.reshape(gts, (-1, n_ill, gts.shape[-1] // n_ill))

    mask = n_ill * mask - 1
    mask = tf.cast(mask, float)
    mask = tf.reshape(mask, (-1, n_ill))

    y_pred = tf.reshape(y_pred, (-1, 1, y_pred.shape[-1]))
    # illuminant_losses = illuminant_loss(gts, y_pred)
    illuminant_losses = cosine_similarity(gts, y_pred)
    # illuminant_losses = tf.reduce_mean(tf.abs(gts-y_pred), axis=-1)
    triplet = mask * illuminant_losses
    ls = tf.maximum(0., tf.reduce_sum(triplet, axis=-1) + alpha)

    return ls


class ClusteringIlluminantLoss(Loss):

    def __init__(self, n_ill=2, alpha=0., shrink=2):
        super().__init__()
        self.n = n_ill
        self.a = alpha
        self.shrink = shrink

    def call(self, y_true, y_pred):
        mask = y_true[...,:self.n]
        gts = y_true[...,self.n:]
        ls = cluster_illuminant_loss(mask, gts, y_pred, self.n, self.a)

        return tf.reduce_mean(ls)



if __name__ == '__main__':

    il = IlluminantLoss()
    gt1 = tf.ones((2,9))
    gt2 = tf.ones((2,3)) / 2
    gt2 = tf.concat([gt1[...,:6], gt2], axis = -1)
    loss = il(gt1, gt2)
    l1 = il(gt1[...,:3], gt2[...,:3])
    l2 = il(gt1[...,3:6], gt2[...,3:6])
    l3 = il(gt1[...,6:9], gt2[...,6:9])
    l = (l1 + l2 + l3) / 3

    dataset_path = '/media/donik/Disk/Cube2/'
    list_name = 'list_realworld.txt'

    image_paths = Cube2Dataset.load_paths(dataset_path, list_name)
    image_paths1 = list(filter(lambda x: x.find('canon') != -1, image_paths))
    image_paths2 = list(filter(lambda x: x.find('nikon') != -1, image_paths))
    image_paths3 = list(filter(lambda x: x.find('pixel') != -1, image_paths))

    ip = [  # ("canon", image_paths1),
        ("nikon", image_paths2),
        # ("pixel", image_paths3),
    ]

    def normalize_brignthess(img):
        img = img / tf.reduce_mean(tf.linalg.norm(img, axis=-1))
        return img

    with tf.device('/device:CPU:0'):
        for name, image_paths in ip:
            for image_path in image_paths:
                gt = Cube2Dataset.get_gt(image_path)
                img = Cube2Dataset.get_image(image_path, 'img.png', 256, 512, scaling=4)
                mask = Cube2Dataset.get_image(image_path, 'gt_mask.png', 256, 512, scaling=1)
                ls = cluster_illuminant_loss(tf.expand_dims(mask, axis=0), tf.expand_dims(gt, axis=0), tf.expand_dims(img, axis=0))

    # loss_o = GaussCosineLoss(apply_softmax=True, out_shape=(2, 256, 256), center_u=1., center_v=1.5)
    # # loss = CECosineLoss(apply_softmax=True, out_shape=(2, 200, 256))
    # a = tf.zeros((2, 256, 256))
    # b = tf.random.normal((2, 256, 256), 0, 1)
    # a = tf.reshape(a, (-1, 256*256))
    # b = tf.reshape(b, (-1, 256*256))
    # # val = loss(a, b)
    # val_o = loss_o(a, b)

    # print(val)
    # depth = 256
    # color_table = histogram.color_table(depth, center_u=0, center_v=0)
    # color_table_flat = tf.reshape(color_table, (-1, 3))
    # l = tf.constant([[0.7038911, 0.0952613, 0.7038911], [0.11941194, 0.7786637,  0.6159737]])
    # b = tf.concat([1, depth ** 2], axis=0)
    # rgb_true = tf.tile(l, b)
    # rgb_shape = tf.concat([(2, depth**2), tf.constant([3])], axis=0)
    # rgb_true = tf.reshape(rgb_true, rgb_shape)
    #
    # b = tf.concat([(2,), tf.constant([1])], axis=0)
    # color_table_flat = tf.tile(color_table_flat, b)
    # c_shape = tf.concat([(2,), tf.constant([depth * depth, 3])], axis=0)
    # color_table_flat = tf.reshape(color_table_flat, c_shape)
    # cos = cosine_similarity(rgb_true, color_table_flat)
    # cos2 = tf.sqrt(cos)
    # import visualizer
    # cos_vis, cos2_vis = tf.reshape(cos, (2, depth, depth)), tf.reshape(cos2, (2, depth, depth))
    # color_table = tf.reshape(color_table_flat, (2, depth, depth, 3))
    # cos_vis1 = 2 / (1/cos_vis[0] + 1/cos_vis[1])
    # cos_vis2 = ((cos_vis[0] + cos_vis[1]) / 2)
    # cos_vis = tf.stack((cos_vis1, cos_vis2), axis=0)
    # # cos_vis = tf.tile(cos_vis, [2,1,1])
    # visualizer.visualize([visualizer.create_mask(l[0], (depth, depth))])
    # visualizer.visualize(cos_vis)
    # # visualizer.visualize(cos_vis2)
    # visualizer.visualize(color_table)
    # visualizer.visualize(cos_vis[:,:,:,tf.newaxis] * color_table)


