import sys
import random

from cart import build_tree, predictions_with_tree
from bagging import create_ensemble, predictions_with_ensemble
from gviz_helpers import tree_to_pdf
from utils import train_validation_split, accuracy


def main(args):
    def to_f(vals): return [float(x) for x in vals]

    random.seed(1)

    with open('dataforcshc.csv', 'r') as fp:
        instances = [to_f(line.split(',')) for line in fp.readlines()[1:]]

    train, val = train_validation_split(instances, train_split=0.9, shuffle=True)
    true_ys = [instance[-1] for instance in val]

    use_ensemble = True

    if not use_ensemble:
        clf = build_tree(train)
        pred_ys = predictions_with_tree(clf, val)
        print(f'Accuracy: {accuracy(true_ys, pred_ys)}')
        tree_to_pdf(clf, f'meinbaum')
    else:
        clf = create_ensemble(10, train, num_features=20, max_depth=10)
        pred_ys = predictions_with_ensemble(clf, val)
        print(f'Accuracy with ensemble: {accuracy(true_ys, pred_ys)}')
        for ix, tree in enumerate(clf):
            pred_ys = predictions_with_tree(tree, val)
            print(f'Accuracy with tree {ix+1}: {accuracy(true_ys, pred_ys)}')
            tree_to_pdf(tree, f'meinbaum_{ix+1}')


if __name__ == '__main__':
    main(sys.argv)
