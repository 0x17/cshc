import sys
import random
import utils

from cart import build_tree, predictions_with_tree, path_of_instance
from bagging import create_ensemble, predictions_with_ensemble
from gviz_helpers import tree_to_pdf
from utils import train_validation_split, accuracy


def main(args):
    utils.verbose = True

    def to_f(vals): return [float(x) for x in vals]

    random.seed(23)

    with open('dataforcshc.csv', 'r') as fp:
        instances = [to_f(line.split(',')) for line in fp.readlines()[1:]]

    train, val = train_validation_split(instances, train_split=0.9, shuffle=True)
    test = val
    true_ys = [instance[-1] for instance in test]

    use_ensemble = True

    if not use_ensemble:
        clf = build_tree(train, max_depth=9999)
        pred_ys = predictions_with_tree(clf, test)

        for ix, pred_y in enumerate(pred_ys):
            if pred_y != true_ys[ix]:
                print(f'Instance incorrect {ix}: path {path_of_instance(clf, test[45])}')

        print(f'Accuracy: {accuracy(true_ys, pred_ys)}')
        tree_to_pdf(clf, f'meinbaum')
    else:
        clf = create_ensemble(10, train, num_features=50, max_depth=9999, subsample_size=500)
        pred_ys = predictions_with_ensemble(clf, test)
        print(f'Accuracy with ensemble: {accuracy(true_ys, pred_ys)}')
        for ix, tree in enumerate(clf):
            pred_ys = predictions_with_tree(tree, test)
            print(f'Accuracy with tree {ix+1}: {accuracy(true_ys, pred_ys)}')
            #tree_to_pdf(tree, f'meinbaum_{ix+1}')


if __name__ == '__main__':
    main(sys.argv)
