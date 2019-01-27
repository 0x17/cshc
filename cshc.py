import sys
import random
from gviz_helpers import tree_to_pdf


def split(instances, feature_ix, split_value):  # set is matrix of [xs...y] row vectors
    lsubset = [instance for instance in instances if instance[feature_ix] < split_value]
    rsubset = [instance for instance in instances if instance[feature_ix] >= split_value]
    return {'feature_ix': feature_ix,
            'value': split_value,
            'l': lsubset,
            'r': rsubset,
            'cost': nmisclass(lsubset) + nmisclass(rsubset)}


def nmisclass(subset):
    card = len(subset)
    csum = sum(instance[-1] for instance in subset)  # instance is [xs...y] row vector
    return card - csum if csum > card / 2 else csum


def possible_splits(instances):
    nfeatures = len(instances[0]) - 1
    for feature_ix in range(nfeatures):
        for instance in instances:
            val = instance[feature_ix]
            yield feature_ix, val


def cheapest_split(instances):
    best_split = {
        'feature_ix': 0,
        'value': 0,
        'l': [],
        'r': [],
        'cost': -1
    }
    for feature_ix, val in possible_splits(instances):
        res = split(instances, feature_ix, val)
        if best_split['cost'] == -1 or res['cost'] < best_split['cost']:
            best_split = res
    return best_split


class Node:
    def __init__(self, feature_ix=0, value=0, left=None, right=None):
        self.feature_ix = feature_ix
        self.value = value
        self.left = left
        self.right = right

    def __str__(self):
        lstr = self.left if isinstance(self.left, float) else 'lchild'
        rstr = self.right if isinstance(self.right, float) else 'rchild'
        return f'Node({self.feature_ix},{self.value},{lstr},{rstr})'


def dom_class(instances):
    return 1.0 if sum(inst[-1] for inst in instances) > len(instances) / 2 else 0.0


def build_tree(instances, depth=0, max_depth=5):
    res = cheapest_split(instances)
    lsubtree = build_tree(res['l'], depth + 1, max_depth) if len(res['l']) >= 10 and depth < max_depth and res['cost'] > 0.0 else dom_class(res['l'])
    rsubtree = build_tree(res['r'], depth + 1, max_depth) if len(res['r']) >= 10 and depth < max_depth and res['cost'] > 0.0 else dom_class(res['r'])
    return Node(res['feature_ix'], res['value'], lsubtree, rsubtree)


def predict_with_tree(root_node, instance):
    if instance[root_node.feature_ix] < root_node.value:
        if isinstance(root_node.left, float):
            return root_node.left
        else:
            return predict_with_tree(root_node.left, instance)
    else:
        if isinstance(root_node.right, float):
            return root_node.right
        else:
            return predict_with_tree(root_node.right, instance)


def main(args):
    def to_f(vals): return [float(x) for x in vals]

    with open('iris.csv', 'r') as fp:
        instances = [to_f(line.split(',')) for line in fp.readlines()[1:]]

    random.shuffle(instances)
    tree = build_tree(instances)
    # pred_y = predict_with_tree(tree, instances[3])
    # true_y = instances[3][-1]
    tree_to_pdf(tree, 'meinbaum')


if __name__ == '__main__':
    main(sys.argv)
