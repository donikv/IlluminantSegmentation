import tensorflow as tf
from general.processing import histogram


def iou_bb(y_true, y_pred, epsilon=1e-10):
    '''
    Computes IoU of the given bounding boxes. Box coordinates should be given in the order x1y1 x2y2.

    :param y_true: First bounding box, [B, 2, 2], [2, 2], [B, 4], [4,]
    :param y_pred: Second bounding box, [B, 2, 2], [2, 2], [B, 4], [4,]
    :return: IoU measure, [B,], []
    '''

    is_batched = len(y_true.shape) == 3

    y_true = tf.reshape(y_true, (-1, 2, 2))
    y_pred = tf.reshape(y_pred, (-1, 2, 2))

    mins = tf.maximum(y_true[:, 0, :], y_pred[:, 0, :]) # [B, 2]
    maxs = tf.minimum(y_true[:, 1, :], y_pred[:, 1, :]) # [B, 2]

    intersect = tf.maximum(0, maxs - mins)
    i_area = tf.reduce_prod(intersect, axis = -1)

    t_area = tf.reduce_prod(y_true[:, 1, :] - y_true[:, 0, :], axis = -1)
    p_area = tf.reduce_prod(y_pred[:, 1, :] - y_pred[:, 0, :], axis = -1)
    u_area = tf.cast(t_area + p_area - i_area, dtype=float)
    i_area = tf.cast(i_area, dtype=float)

    iou = i_area / (u_area + epsilon)

    if not is_batched:
        return tf.squeeze(iou)

    return iou


@tf.function
def dice_coef(y_true, y_pred, num_classes=3, smooth=1e-7, ignore_black=False):
    '''
    Dice coefficient for 3 categories. Ignores background pixel label 1
    Pass to model as metric during compile statement
    '''
    y_true_f = tf.reshape(y_true[...,:num_classes], (-1, num_classes))
    y_pred_f = tf.reshape(y_pred[...,:num_classes], (-1, num_classes))
    if ignore_black:
        not_blck = tf.argmax(y_true_f, axis=-1) != tf.constant(1, dtype=tf.int64 )
        y_true_f = y_true_f[not_blck]
        y_pred_f = y_pred_f[not_blck]
        y_true_f = tf.stack((y_true_f[...,0], y_true_f[...,2]))
        y_pred_f = tf.stack((y_pred_f[..., 0], y_pred_f[..., 2]))
    intersect = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    denom = tf.reduce_sum(y_true_f + y_pred_f, axis=0)
    dice = (2. * (intersect) / (denom + smooth))
    return dice


def dice_coef_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef(y_true, y_pred)

class Dice(tf.keras.metrics.Metric):

    def __init__(self, is_class_invariant=False, num_classes=3, ignore_black=False):
        super(Dice, self).__init__(name='dice_acc')
        self.is_invariant = is_class_invariant
        self.ignore_black = ignore_black
        # self.dice = tf.zeros(1)
        self.total_dice = self.add_weight("total", shape=[num_classes], initializer="zeros")
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        dc = dice_coef(y_true, y_pred, num_classes=self.num_classes, ignore_black=self.ignore_black)
        self.total_dice.assign(dc)
        return self.total_dice

    def result(self):
        dice = self.total_dice
        dice = tf.reduce_mean(dice)
        if self.is_invariant:
            dice = tf.maximum(dice, 1 - dice)

        return dice

    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

@tf.function
def cosine_similarity(l1, l2, eps=1e-7, keepdims=False):
    l1_norm = tf.norm(l1, axis=-1, keepdims=True) + eps
    l2_norm = tf.norm(l2, axis=-1, keepdims=True) + eps
    l1l2 = tf.reduce_sum(l1*l2, axis=-1, keepdims=True)
    cos = (l1l2) / (l1_norm*l2_norm)
    cos = tf.clip_by_value(cos, -1, 1)
    return tf.squeeze(tf.acos(cos)) if not keepdims else tf.acos(cos)


def cos_sim(y_true, y_pred):
    return cosine_similarity(y_true, y_pred) * 180 / 3.14


def multi_cos_sim(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1,3))
    y_pred = tf.reshape(y_pred, (-1,3))
    return tf.reduce_mean(cosine_similarity(y_true, y_pred) * 180 / 3.14)


def binned_cosine_similarity(y_true, y_pred, depth):
    if y_true.shape[-1] != depth ** 2:
        uv_true = histogram.decode_bins(y_true, depth)
    else:
        uv_true = y_true
    uv_pred = histogram.decode_bins(y_pred, depth)
    if uv_true.shape[-1] != 3:
        rgb_true = histogram.from_uv(uv_true)
    else:
        rgb_true = uv_true
    rgb_pred = histogram.from_uv(uv_pred)
    return -1. * tf.keras.losses.cosine_similarity(rgb_true, rgb_pred)


class BinnedCosineSimilarity(tf.keras.metrics.Metric):

    def __init__(self, depth=16, bs=50):
        super(BinnedCosineSimilarity, self).__init__(name='Binned_cosine_similarity')
        # self.dice = tf.zeros(1)
        self.total_cos_sim = self.add_weight("total", shape=[bs], initializer="zeros")
        self.depth = depth

    def update_state(self, y_true, y_pred, sample_weight=None):
        sim = binned_cosine_similarity(y_true, y_pred, self.depth)
        self.total_cos_sim.assign(sim)
        return self.total_cos_sim

    def result(self):
        cos = self.total_cos_sim
        cos = tf.reduce_mean(cos)
        return cos

    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))


if __name__ == '__main__':
    gt1 = tf.ones((2,9))
    gt2 = tf.ones((2,4)) / 2
    gt2 = tf.concat([gt1[...,:5], gt2], axis = -1)
    cos = multi_cos_sim(gt1, gt2)

    a = tf.constant([[145, 155], [145, 155]])
    b = tf.constant([[11, 12], [11, 12]])
    ab = tf.stack((b,a), axis=-1)
    ba = tf.stack((b,a), axis=1)
    print(iou_bb(ab, ba))
#
# ill1 = tf.constant([0.6, 0.2, 0.1])
# ill2 = tf.constant([0.1, 0.7, 0.1])
# ill1n = ill1 / tf.norm(ill1)
# ill2n = ill2 / tf.norm(ill2)
#
# depth = 16
# cos = -1. * tf.keras.losses.cosine_similarity(ill1, ill2)
# bin1 = histogram.encode_bins(histogram.bin(ill1, depth)[0], depth=depth)
# bin2 = histogram.encode_bins(histogram.bin(ill2, depth)[0], depth=depth)
# bin_cos = binned_cosine_similarity(bin1, bin2, depth)
# print(f"cos:{cos}, binned_cos:{bin_cos}")

