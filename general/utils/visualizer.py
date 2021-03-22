import math

import matplotlib.pyplot as plt
import tensorflow as tf


def visualize(images, custom_transform=lambda x: x, title=None, titles=None, out_file=None, in_line=False,cb=True):
    """
    Plot image and masks.
    If two pairs of images and masks are passes, show both.
    """

    rows = math.ceil((len(images) / 2))
    cols = 2 if len(images) > 1 else 1
    if in_line:
        f, ax = plt.subplots(1, len(images), figsize=(30, 30), squeeze=True)
    else:
        f, ax = plt.subplots(rows, cols, figsize=(30, 30), squeeze=False)
    f.tight_layout()
    if title is not None:
        f.suptitle(str(title), fontsize=64)
    for idx, img in enumerate(images):
        if in_line:
            cur_ax = ax[idx]
        else:
            cur_ax = ax[int(idx / 2)][idx % 2]
            im = custom_transform(img)
        pcm = cur_ax.imshow(im)
        cur_ax.axis('off')
        if len(im.shape) < 3 and cb:
            f.colorbar(pcm, ax=cur_ax)
        if titles is not None and len(titles) > idx:
            cur_ax.set_title(titles[idx], fontweight="bold", size=40, loc='left')

    if out_file is not None:
        plt.savefig(out_file, bbox_inches='tight')
    plt.show()


def create_mask(mask, sz=None):
    if sz is None:
        sz = mask.shape
    mask = tf.broadcast_to(mask, (sz[0], sz[1], mask.shape[-1]))
    return mask