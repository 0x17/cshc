import itertools

from joblib import Parallel, delayed
from utils import log


def split(instances, feature_ix, split_value):  # set is matrix of [xs...y] row vectors
    lsubset = [instance for instance in instances if instance[feature_ix] < split_value]
    rsubset = [instance for instance in instances if instance[feature_ix] >= split_value]
    return {'feature_ix': feature_ix,
            'value': split_value,
            'l': lsubset,
            'r': rsubset}


def split_cost(instances, feature_ix, split_value):
    '''def num_misclassifications(func):
        in_subset = lambda instance: func(instance[feature_ix] < split_value)
        card, csum = functools.reduce(lambda acc, instance: (acc[0] + (1 if in_subset(instance) else 0), acc[1] + (instance[-1] if in_subset(instance) else 0)), instances, (0, 0))
        return card - csum if csum > card / 2 else csum
    return num_misclassifications(lambda x: x) + num_misclassifications(lambda x: not x)'''

    sleft, sright = 0, 0
    sleft_card, sright_card = 0, 0
    for instance in instances:
        if instance[feature_ix] < split_value:
            sleft += instance[-1]
            sleft_card += 1
        else:
            sright += instance[-1]
            sright_card += 1
    nmisclass_left = sleft_card - sleft if sleft > sleft_card / 2 else sleft
    nmisclass_right = sright_card - sright if sright > sright_card / 2 else sright
    return nmisclass_left + nmisclass_right


def possible_splits(instances):
    nfeatures = len(instances[0]) - 1
    for feature_ix in range(nfeatures):
        vals = []
        for instance in instances:
            val = instance[feature_ix]
            if val not in vals:
                vals.append(val)
                yield feature_ix, val


def possible_splits_for_index(instances, feature_ix):
    vals = []
    for instance in instances:
        val = instance[feature_ix]
        if val not in vals:
            vals.append(val)
            yield feature_ix, val


def cheapest_split(instances, parallel=True):
    log(f'Cheapest split for {len(instances)} instances...')

    if parallel:
        def compute_split_cost(feature_ix, val):
            return feature_ix, val, split_cost(instances, feature_ix, val)

        def compute_split_costs_for_feature(feature_ix):
            log('Comp split for feature'+str(feature_ix))
            return [compute_split_cost(feature_ix, val) for feature_ix, val in
                    possible_splits_for_index(instances, feature_ix)]

        results = Parallel(n_jobs=-1)(delayed(compute_split_costs_for_feature)(feature_ix) for feature_ix in range(len(instances[0]) - 1))
        results = [ elem for l in results for elem in l ]
        mincost = min(res[2] for res in results)
        best_split = next(res for res in results if res[2] == mincost)
        best_cost = mincost
    else:
        best_split = (0, 0)
        best_cost = -1

        for feature_ix, val in possible_splits(instances):
            cost = split_cost(instances, feature_ix, val)
            if best_cost == -1 or cost < best_cost:
                best_cost = cost
                best_split = (feature_ix, val)
                log('New split incumbent ' + str(best_split) + ' with cost ' + str(best_cost))

    res = split(instances, best_split[0], best_split[1])
    res['cost'] = best_cost
    log('Found split ' + str(best_split) + ' with cost ' + str(best_cost))
    return res


class Node:
    def __init__(self, feature_ix=0, value=0, left=None, right=None, cost=0.0):
        self.feature_ix = feature_ix
        self.value = value
        self.left = left
        self.right = right
        self.cost = cost

    def __str__(self):
        lstr = self.left if isinstance(self.left, float) else 'lchild'
        rstr = self.right if isinstance(self.right, float) else 'rchild'
        return f'Node({self.feature_ix},{self.value},{lstr},{rstr},{self.cost})'


def dom_class(instances):
    return 1.0 if sum(inst[-1] for inst in instances) > len(instances) / 2 else 0.0


def build_tree(instances, depth=0, max_depth=5, min_clustersize=0):
    res = cheapest_split(instances, parallel=False)

    if res['cost'] == 0: log('Hit zero cost')
    if depth >= max_depth: log('Hit max depth')

    if res['cost'] == 0.0:
        dcl, dcr = dom_class(res['l']), dom_class(res['r'])
        if dcl == dcr: return dcl
        return Node(res['feature_ix'], res['value'], dcl, dcr, 0)

    lsize, rsize = len(res['l']), len(res['r'])
    if lsize == 0:
        return dom_class(res['r'])
    if rsize == 0:
        return dom_class(res['l'])

    lsubtree, rsubtree = dom_class(res['l']) if lsize < min_clustersize or depth >= max_depth else build_tree(res['l'],
                                                                                                              depth + 1,
                                                                                                              max_depth), \
                         dom_class(res['r']) if rsize < min_clustersize or depth >= max_depth else build_tree(res['r'],
                                                                                                              depth + 1,
                                                                                                              max_depth)

    return Node(res['feature_ix'], res['value'], lsubtree, rsubtree, res['cost'])


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


def predictions_with_tree(root_node, instances):
    return [predict_with_tree(root_node, instance) for instance in instances]


def path_of_instance(root_node, instance):
    if instance[root_node.feature_ix] < root_node.value:
        if isinstance(root_node.left, float):
            return f'(value_left={root_node.left})'
        else:
            return 'left; ' + path_of_instance(root_node.left, instance)
    else:
        if isinstance(root_node.right, float):
            return f'(value_right={root_node.right})'
        else:
            return 'right; ' + path_of_instance(root_node.right, instance)
