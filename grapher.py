import matplotlib.pyplot as plt
import numpy as np
import random
import copy

def show_figures(images, title):
    plt.figure(figsize=(8, 10))
    for idx, val in enumerate(images):
        plt.subplot(len(11) / 5 + 1, 5, idx + 1)
        plt.subplots_adjust(top=0.99, bottom=0.01, hspace=1.5, wspace=0.4)
        plt.xticks([], [])
        plt.yticks([], [])
        im, im_class, im_dist = val
        plt.title("{} {}".format(im_class, im_dist))
        plt.imshow(im, cmap="gray")
    plt.tight_layout()
    plt.savefig(title)