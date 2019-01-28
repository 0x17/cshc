import math
import random


def train_validation_split(samples, train_split=0.5, shuffle=True):
    sc = samples.copy()
    if shuffle: random.shuffle(sc)
    scl = len(sc)
    split_index = int(math.floor(scl * train_split))
    return sc[:split_index + 1], sc[split_index + 1:]


def accuracy(true_ys, pred_ys):
    return sum(1 if true_y == pred_ys[ix] else 0 for ix, true_y in enumerate(true_ys)) / len(true_ys)
