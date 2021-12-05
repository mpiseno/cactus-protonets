import collections
import numpy as np


def compute_cluster_modes(cluster_to_idxs, gt_labels):
        # map mode -> [cluster labels with this mode]
        cluster_modes = {}
        for cluster_label, idxs in cluster_to_idxs.items():
            labels_this_cluster = gt_labels[idxs]
            counts = collections.Counter(labels_this_cluster)
            mode = counts.most_common(1)
            cluster_modes[cluster_label] = mode

        return cluster_modes


def cluster_quality(cluster_labels, gt_labels):
    cluster_to_idxs = collections.defaultdict(list)
    for idx, cluster_label in enumerate(cluster_labels):
        cluster_to_idxs[cluster_label].append(idx)

    cluster_modes = compute_cluster_modes(cluster_to_idxs, gt_labels)
    num_mislabeled = 0
    for cluster_label, idxs in cluster_to_idxs.items():
        mode = cluster_modes[cluster_label]
        labels = gt_labels[idxs]
        num_mislabeled += sum([mode != label for label in labels])

    mislabeled_ratio = num_mislabeled / len(gt_labels)
    print(f'mislabeled ratio: {mislabeled_ratio} | ({num_mislabeled} / {len(gt_labels)})')
    return mislabeled_ratio