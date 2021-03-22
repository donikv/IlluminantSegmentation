import tensorflow as tf
import tensorflow.keras.layers as l
import numpy as np


class DiffHist(l.Layer):

    def __init__(self, b, range_init=(0., 1.), w_init=None, weighted=False, **kwargs):
        super().__init__(**kwargs)
        self.b = b
        self.range_init = range_init
        self.w_init = 1/b if w_init is None else w_init
        self.weighted = weighted

    def build(self, input_shape):
        k = input_shape[-1] if not self.weighted else input_shape[-1] - 1
        b = self.b

        start, stop = self.range_init
        mi_k = tf.range(start, stop, delta=(stop - start) / b)
        mi_kb = tf.tile(mi_k, (k,))
        mi_kb = tf.reshape(mi_kb, (k,b))
        # mi_kb = tf.tile(mi_kb, (h * w, 1))
        self.mi_kb = tf.Variable(mi_kb, trainable=self.trainable, name='centers')

        w_kb = tf.ones((k, b)) * self.w_init
        self.w_kb = tf.Variable(w_kb, trainable=self.trainable, name='widths')

    def call(self, inputs, **kwargs):
        if self.weighted:
            inputs = inputs[..., 1:]
            wx = inputs[..., 0:1]  # tf.where(inpt[..., 0] < 0.03, 0., 1.)

        input_shape = inputs.shape
        inputs = tf.expand_dims(inputs, axis=-1)
        inputs = tf.tile(inputs, (1, 1, 1, 1, self.b))
        a = inputs - self.mi_kb
        a = tf.abs(a)
        b = 1 - (a / self.w_kb)
        c = tf.maximum(0., b)

        if self.weighted:
            c = c * tf.expand_dims(wx, axis=-1)

        return c


class DiffHist2D(l.Layer):

    def __init__(self, b, range_init_x=(0., 1.), range_init_y=(0., 1.), w_init:tuple=None):
        super().__init__()
        self.b = b
        self.range_init_x = range_init_x
        self.range_init_y = range_init_y
        self.w_init = (1/b, 1/b) if w_init is None else w_init

    def build(self, input_shape):
        # k = input_shape[-1]
        b = self.b

        start, stop = self.range_init_x
        mi_k = tf.range(start, stop, delta=(stop - start) / b)
        mi_k_x, mi_k_y = tf.meshgrid(mi_k, mi_k)
        mi_k_x = tf.reshape(mi_k_x, (b*b, ))
        mi_k_y = tf.reshape(mi_k_y, (b*b, ))
        self.mi_kx = tf.Variable(mi_k_x, trainable=True, name='centers')
        self.mi_ky = tf.Variable(mi_k_y, trainable=True, name='centers')

        w_kb = tf.ones((b*b, )) / b
        self.w_b = tf.Variable(w_kb, trainable=True, name='widths')

    def call(self, inputs, **kwargs):
        inputs = tf.expand_dims(inputs, axis=-1)
        inputs = tf.tile(inputs, (1, 1, 1, 1, self.b * self.b))
        ax = tf.abs(inputs[...,1,:] - self.mi_kx)
        bx = 1 - (ax / self.w_b)
        cx = tf.keras.activations.relu(bx)
        ay = tf.abs(inputs[...,2,:] - self.mi_ky)
        by = 1 - (ay / self.w_b)
        cy = tf.keras.activations.relu(by)

        return cx * cy * inputs[...,0,:]


def plot_histogram(h:DiffHist):
    import matplotlib.pyplot as plt
    from matplotlib import collections as mc
    miss = h.mi_kb
    wss = h.w_kb

    f, axes = plt.subplots(miss.shape[0], 1)
    c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])
    for i, (mis, ws) in enumerate(zip(miss.numpy(), wss.numpy())):
        lines = []
        colors = []
        for j, (mi, w) in enumerate(zip(mis, ws)):
            x0 = mi - w
            x1 = mi
            x2 = mi + w
            color = c[j % c.shape[0]]
            lines.append([(x0,0), (x1,1)])
            lines.append([(x1,1), (x2,0)])
            colors.append(color)
            colors.append(color)
        lc = mc.LineCollection(lines, 2, colors=colors)
        if miss.shape[0] > 1:
            ax = axes[i]
        else:
            ax = axes
        ax.add_collection(lc)
        ax.autoscale()

        ax.margins(0.1)
    plt.show()

def build_simple_model(h, w, n, c, b, range_init, w_init, temp=1.):
    input = l.Input((n,h,w,c))
    hist = DiffHist(b, range_init, w_init)

    #Handle multiple preprocessed images
    xs = []
    for i in range(n):
        x = hist(input[:, i, :, :, 1:])
        x = x * tf.expand_dims(tf.expand_dims(input[:, i, :, :, 0], axis=-1), axis=-1)
        x = tf.sqrt(x)
        xs.append(x)
    x = tf.concat(xs, axis=-2)

    y = tf.reduce_mean(x, axis=[1, 2])
    y = tf.math.divide_no_nan(y, tf.reduce_sum(y, axis=-1, keepdims=True))
    # y = tf.reshape(y, (-1, np.prod(y.shape[1:])))

    y = tf.reduce_sum(y, axis=-2)
    conv = l.Dense(256, use_bias=False, activation='linear')
    y = conv(y)
    y = l.BatchNormalization()(y)
    # ys = []
    # for i in range(n):
    #     x = y[:,i,:]
    #     # x = x[:, :, tf.newaxis]
    #     ys.append(conv(x))
    # y = tf.stack(ys, axis=-1)
    # y = tf.reduce_sum(y, axis=-1)
    y = tf.exp(y/temp)

    return tf.keras.Model(input, y), hist


def build_model(h, w, c, b, range_init, w_init, out, activation='linear'):
    input = l.Input((h, w, c))
    hist = DiffHist(b, range_init, w_init)

    x = hist(input[..., 1:])
    x = x * tf.expand_dims(tf.expand_dims(input[..., 0], axis=-1), axis=-1)
    x = tf.sqrt(x)

    y = tf.reduce_mean(x, axis=[1, 2])
    y = tf.math.divide_no_nan(y, tf.reduce_sum(y, axis=-1, keepdims=True))
    y1 = y[..., 0:(c-1)//2, :]
    y2 = y[..., (c-1)//2:c-1, :]

    def dense_encode(y1):
        y1 = tf.reshape(y1, (-1, np.prod(y1.shape[1:])))
        y1 = l.Dense(b//2, use_bias=False)(y1)  # TODO: Dodati mogucnost duplog histograma, duplog dense layera
        y1 = l.BatchNormalization()(y1)
        y1 = l.Dropout(0.2)(y1)
        y1 = l.ReLU()(y1)
        return y1

    y1 = dense_encode(y1)
    y2 = dense_encode(y2)
    y = l.concatenate((y1, y2))

    y = l.Dense(out, activation=activation)(y)
    y = l.Softmax()(y)

    return tf.keras.Model(input, y), hist

def hist_coss_loss(mis, rgb_true, bs):
    hsvs = tf.stack([mis, tf.ones_like(mis), tf.ones_like(mis)], axis=-1)
    rgbs = tf.image.hsv_to_rgb(hsvs)
    rgbs = tf.tile(rgbs, (bs, 1, 1)) # bs x b x 3
    rgb_true = tf.reshape(tf.tile(rgb_true, [1, mis.shape[-1]]), rgbs.shape)
    cos = losses.cosine_similarity(rgb_true, rgbs) # bs x b
    return cos * 180 / 3.14


def hist_loss(y_pred, y_true, mis, bs):
    cos_loss = tf.sqrt(hist_coss_loss(mis, y_true, bs))
    return y_pred * cos_loss


class HistLoss1D(tf.losses.Loss):

    def __init__(self, bs, hist:DiffHist, sum=True):
        super().__init__()
        self.bs = bs
        self.hist = hist
        self.sum = sum

    def call(self, y_true, y_pred):
        # y_pred = tf.squeeze(y_pred, axis=-2)
        if self.sum:
            y_pred = y_pred / (tf.reduce_sum(y_pred, axis=-1, keepdims=True) + 1e-10)
        return hist_loss(y_pred, y_true, self.hist.mi_kb, self.bs)


class HistSimilarity():

    def __init__(self, bs, hist:DiffHist):
        self.bs = bs
        self.hist=hist

    __name__ = 'hist_cos_similarity'

    def __call__(self, y_true, y_pred):
        mi_idx = tf.argmax(y_pred, axis=-1)
        mi = tf.gather(self.hist.mi_kb, mi_idx, axis=-1)
        mi = tf.reshape(mi, (1, -1, 1))
        mi_hsv = tf.concat([mi, tf.ones_like(mi), tf.ones_like(mi)], axis=-1)
        mi_rgb = tf.image.hsv_to_rgb(mi_hsv)[0]
        coss = losses.cosine_similarity(mi_rgb, y_true) * 180 / 3.14
        return coss
        # return tf.reduce_mean(coss, axis=-1)


if __name__ == '__main__':
    from general.datasets import Cube2Dataset
    from general.processing import histogram
    from general.training import losses
    from general.utils import visualizer as v
    import matplotlib.pyplot as plt
    import scipy.signal as s

    def moving_avg(x, n):
        mv = np.convolve(x, np.ones(n) / n, mode='valid')
        # return mv
        return np.concatenate(([0 for k in range(n - 1)], mv))

    with tf.device('/device:cpu:0'):
        img_path = 'D:/fax/Cube2/outdoor/canon_550d/outdoor1/4'
        inpt = tf.ones((1, 100, 100, 5))
        inpt2 = tf.random.uniform((1, 100, 100, 5))
        inpt = tf.concat((inpt, inpt2), axis=0)
        d = 256
        hist = DiffHist(d, (0, 1), w_init=1 / d)
        inpt = Cube2Dataset.get_image(img_path, 'img.png', 256, 512)
        inpt = tf.expand_dims(inpt, axis=0)
        # inpt = dp.__process_images__(inpt, [1, 4])
        inpt_rg = tf.math.divide_no_nan(inpt[..., 0], inpt[..., 1])
        inpt_bg = tf.math.divide_no_nan(inpt[..., 2], inpt[..., 1])
        inpt_rb = tf.stack((inpt_rg, inpt_bg), axis=-1)
        inpt_uv, Iy = histogram.to_uv(inpt)
        inpt_hsv = tf.image.rgb_to_hsv(inpt)
        inpt_h = inpt_hsv[..., 0:1]

        # inpt_rb = tf.expand_dims(inpt_rb, axis=0)
        # inpt_rb = tf.concat((inpt_rb[:,0,:,:,:], inpt_rb[:,1,:,:,:]), axis=-1)
        # Iy = tf.transpose(Iy, (1, 2, 0))

        Iy = tf.where(Iy < 0.05, 0., Iy)
        w = tf.where(Iy == 0., Iy, tf.ones_like(Iy))
        y = hist(inpt_h)
        y = y * tf.expand_dims(tf.expand_dims(w, axis=-1), axis=-1)
        y = tf.reduce_sum(y, axis=[1,2])
        y = tf.sqrt(y / tf.reduce_sum(y, axis=-1, keepdims=True))
        rb = tf.argmax(y, axis=-1)
        v.visualize(y)
        v.visualize(inpt)

        #CCC LOSS FOR SINGLE HISTOGRAM
        rgb_true = tf.convert_to_tensor([[0., 0., 1.], [0., 1., 0.]])
        loss = HistLoss1D(bs=2, hist=hist)
        yb = tf.concat([y,y], axis=0)
        yb = yb / (tf.reduce_sum(yb) + 1e-10)
        cos = loss(rgb_true, y)
        sim = HistSimilarity(2, hist)
        cos_sim = sim(rgb_true, yb)
        # v.visualize([cos])

        # y1 = moving_avg(y[0,0], 10)
        # y = y1[np.newaxis, np.newaxis, :]

        x = np.arange(0., 1., 1 / d)
        plt.bar(x, yb[0,0], width=1 / d)

        #HISTOGRAM PEAKS
        peaks, props = s.find_peaks(yb[0,0], distance=18)
        peak_heights = np.array(list(map(lambda x: yb[0, 0, x], peaks)))
        pph = sorted(zip(peaks, peak_heights), key=lambda x: x[1], reverse=True)
        peaks = np.array(list(map(lambda x: x[0], pph)))
        colors = ["red", "green", "yellow", "cyan", "magenta"]
        ills = []
        for p, c in zip(peaks, colors):
            p1 = yb[0, 0, p]
            ys = np.arange(0., p1, p1/100)[:100]
            plt.plot(np.ones(100) * p / d, ys, color=c)
            ill = v.create_mask(tf.convert_to_tensor([p / d, 1, 1]), [10, 10])
            ills.append(tf.image.hsv_to_rgb(ill))
        # inp = inpt_rg[Iy > 0]
        # kde = st.gaussian_kde(dataset=tf.reshape(inp, (-1,)))
        # plt.plot(x, kde.evaluate(x))
        plt.show()
        v.visualize(ills)

        v.visualize([Cube2Dataset.get_image(img_path, 'gt.png', 256, 512)])

        #2D HISTOGRAM
        # plt.bar(x, y[0,1], width=1/64)
        # plt.show()
        # v.visualize(inpt)
        #
        # hist2d = DiffHist2D(64, (-2, 2))
        # w = tf.where(Iy == 0., Iy, tf.ones_like(Iy))
        # img = tf.stack([w, inpt_uv[..., 0], inpt_uv[..., 1]], axis=-1)
        # img = tf.image.resize(img, (256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # y = hist2d(img)
        # y = tf.reduce_mean(y, axis=[1, 2])
        # y = tf.sqrt(y / tf.reduce_sum(y, axis=-1, keepdims=True))
        # y = tf.reshape(y, (1,64,64))
        # v.visualize(y)
        # v.visualize(img)

        print(y)