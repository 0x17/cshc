import sys
import random

from cart import build_tree, predictions_with_tree
from bagging import create_ensemble, predictions_with_ensemble
from gviz_helpers import tree_to_pdf
from utils import train_validation_split, accuracy


def main(args):
    def to_f(vals): return [float(x) for x in vals]

    random.seed(1)

    with open('iris.csv', 'r') as fp:
        instances = [to_f(line.split(',')) for line in fp.readlines()[1:]]

    train, val = train_validation_split(instances, train_split=0.9, shuffle=True)

    #clf = build_tree(train)
    clf = create_ensemble(4, train, num_features=20)

    # pred_y = predict_with_tree(tree, instances[3])
    # true_y = instances[3][-1]

    pred_ys = predictions_with_ensemble(clf, val)
    true_ys = [instance[-1] for instance in val]
    print(f'Accuracy: {accuracy(true_ys, pred_ys)}')

    for ix, tree in enumerate(clf):
        tree_to_pdf(tree, f'meinbaum_{ix+1}')


if __name__ == '__main__':
    main(sys.argv)
