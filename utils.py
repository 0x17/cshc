import math
import random
import sys

verbose = False


def train_validation_split(samples, train_split=0.5, shuffle=True):
    sc = samples.copy()
    if shuffle: random.shuffle(sc)
    scl = len(sc)
    split_index = int(math.floor(scl * train_split))
    return sc[:split_index + 1], sc[split_index + 1:]


def accuracy(true_ys, pred_ys):
    return sum(1 if true_y == pred_ys[ix] else 0 for ix, true_y in enumerate(true_ys)) / len(true_ys)


def nfold_cross_validation_accuracy(train_func, samples, n=5):
    sc = samples.copy()
    random.shuffle(sc)
    scl = len(sc)
    val_size = math.floor(scl / n)
    offset = 0
    accs = []
    for i in range(n):
        print(f'Training fold no. {i+1}...')
        train = sc[0:offset] + sc[offset + val_size:]
        predict_func = train_func(train)
        val = sc[offset:offset + val_size]
        true_ys = [instance[-1] for instance in val]
        pred_ys = [predict_func(instance) for instance in val]
        accs.append(accuracy(true_ys, pred_ys))
        offset += val_size
    return sum(accs) / len(accs)


def log(message):
    if verbose:
        print(message)
        sys.stdout.flush()
