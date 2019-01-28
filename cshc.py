import sys
import random
from gviz_helpers import tree_to_pdf
import functools


def split(instances, feature_ix, split_value):  # set is matrix of [xs...y] row vectors
    lsubset = [instance for instance in instances if instance[feature_ix] < split_value]
    rsubset = [instance for instance in instances if instance[feature_ix] >= split_value]
    return {'feature_ix': feature_ix,
            'value': split_value,
            'l': lsubset,
            'r': rsubset}


def split_cost(instances, feature_ix, split_value):
    def num_misclassifications(func):
        in_subset = lambda instance: func(instance[feature_ix] < split_value)
        card, csum = functools.reduce(lambda acc, instance: (acc[0] + (1 if in_subset(instance) else 0), acc[1] + (instance[-1] if in_subset(instance) else 0)), instances, (0, 0))
        return card - csum if csum > card / 2 else csum

    return num_misclassifications(lambda x: x) + num_misclassifications(lambda x: not x)


def possible_splits(instances):
    nfeatures = len(instances[0]) - 1
    for feature_ix in range(nfeatures):
        for instance in instances:
            val = instance[feature_ix]
            yield feature_ix, val


def cheapest_split(instances):
    best_split = (0, 0)
    best_cost = -1
    for feature_ix, val in possible_splits(instances):
        cost = split_cost(instances, feature_ix, val)
        if best_cost == -1 or cost < best_cost:
            best_cost = cost
            best_split = (feature_ix, val)
    res = split(instances, best_split[0], best_split[1])
    res['cost'] = best_cost
    return res


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
    lsubtree = build_tree(res['l'], depth + 1, max_depth) if len(res['l']) >= 10 and depth < max_depth and res[
        'cost'] > 0.0 else dom_class(res['l'])
    rsubtree = build_tree(res['r'], depth + 1, max_depth) if len(res['r']) >= 10 and depth < max_depth and res[
        'cost'] > 0.0 else dom_class(res['r'])
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
