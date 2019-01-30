import cart
import random
from utils import log

def random_indices(n, ub):
    l = list(range(ub))
    random.shuffle(l)
    return l[:n]


def create_ensemble(num_trees, instances, num_features=-1, max_depth=5, subsample_size=-1):
    feature_count = len(instances[0]) - 1
    instance_count = len(instances)
    num_features = feature_count if num_features == -1 else num_features
    subsample_size = round(instance_count / 10) if subsample_size == -1 else subsample_size
    ensemble = []
    for i in range(num_trees):
        selected_features = random_indices(num_features, feature_count) + [feature_count]
        selected_instances = random_indices(subsample_size, instance_count)
        subsample = [[feature for feature_index, feature in enumerate(instance) if feature_index in selected_features] for instance_index, instance in enumerate(instances) if instance_index in selected_instances]
        ensemble.append(cart.build_tree(subsample, max_depth=max_depth))
        log(f'Tree no. {i+1} built...')
    return ensemble


def predict_with_ensemble(ensemble, instance):
    return round(sum(cart.predict_with_tree(tree, instance) for tree in ensemble) / len(ensemble))


def predictions_with_ensemble(ensemble, instances):
    return [predict_with_ensemble(ensemble, instance) for instance in instances]
