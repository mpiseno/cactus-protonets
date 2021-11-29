import os
import sys
import glob
import ipdb

from functools import partial

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor

from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose
from tqdm import tqdm

import protonets
from protonets.data.base import Partition, TaskLoader, celeba_partitions
from collections import defaultdict, Counter

DATA_DIR  = os.path.join(os.path.dirname(__file__), '../../data')


def measure_dup_sample_rate(partitions, gt_labels, n_way, n_shot, num_samples=1000):
    num_dup = 0
    for _ in range(num_samples):
        i_partition = torch.randint(low=0, high=len(partitions), size=(1,), dtype=torch.int)
        partition = partitions[i_partition]
        sampled_subset_ids = np.random.choice(partition.subset_ids, size=n_way, replace=False)
        sampled_labels = set()
        for subset_id in sampled_subset_ids:
            indices = np.random.choice(partition[subset_id], n_shot, replace=False)
            labels = set(gt_labels[indices])
            if len(sampled_labels.intersection(labels)):
                num_dup += 1
                break
        
            sampled_labels = sampled_labels.union(labels)
    
    dup_rate = num_dup / num_samples
    print(f'{n_way} way {n_shot} shot duplicate rate: {dup_rate}')


def visualize_partition(Z, pre_labels, post_labels, gt_labels):
    from sklearn.manifold import TSNE
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    def show_scatter(tsne_embedding, labels):
        df = pd.DataFrame()
        df["y"] = labels
        df["comp-1"] = tsne_embedding[:,0]
        df["comp-2"] = tsne_embedding[:,1]

        n_colors = len(np.unique(labels))
        temp = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                        palette=sns.color_palette("hls", n_colors),
                        data=df).set(title="T-SNE Projection")
        plt.legend([],[], frameon=False)

    Z_ = TSNE(n_components=2, verbose=1, random_state=123).fit_transform(Z)
    show_scatter(Z_, pre_labels)
    plt.show()
    show_scatter(Z_, post_labels)
    plt.show()
    show_scatter(Z_, gt_labels)
    plt.show()

    import pdb
    pdb.set_trace()

def kmeans_post_process(kmeans, semi_sup_labels):
    def compute_modes(km_to_idxs, ss_labels):
        # map mode -> [cluster labels with this mode]
        mode_to_cluster_labels = defaultdict(list)
        unique_cluster_var = -1
        for cluster_label, idxs in km_to_idxs.items():
            ss_labels_this_cluster = ss_labels[idxs]
            counts = Counter(ss_labels_this_cluster)
            if len(counts) > 1:
                mode1, mode2 = counts.most_common(2)
                ss_label_mode = mode2[0]
                if mode1[0] != -1:
                    ss_label_mode = mode1[0]
                
                mode_to_cluster_labels[ss_label_mode].append(cluster_label)
            else:
                # Otherwise we want to just keep this cluster intact
                mode_to_cluster_labels[unique_cluster_var].append(cluster_label)
                unique_cluster_var -= 1

        return mode_to_cluster_labels

    def merge_clusters(km_to_idxs, mode_to_km):
        new_km_to_idxs = defaultdict(list)
        for mode, clusters in mode_to_km.items():
            for km_label in clusters:
                new_km_to_idxs[mode].extend(
                    km_to_idxs[km_label]
                )

        return new_km_to_idxs

    # first map kmean's label to a list of idxs with that label
    cluster_label_to_idxs = defaultdict(list)
    for idx, cluser_label in enumerate(kmeans.labels_):
        cluster_label_to_idxs[cluser_label].append(idx)
    
    # compute mode of semi-supervised label for each cluster
    mode_to_cluster_labels = compute_modes(cluster_label_to_idxs, semi_sup_labels)

    # Merge clusters that have the same mode
    new_clusters_to_idxs = merge_clusters(cluster_label_to_idxs, mode_to_cluster_labels)

    new_labels = np.zeros(len(kmeans.labels_))
    for cluster_label, idxs in new_clusters_to_idxs.items():
        new_labels[idxs] = cluster_label

    kmeans.labels_ = new_labels
    return kmeans


def get_partitions_kmeans(
    encodings,
    semi_sup_labels, # unlabeled points will have label -1
    gt_labels,
    n_way, n_shot, n_query,
    random_scaling=True,
    n_partitions=100, n_clusters=500
):
    import os
    os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'  # default runs out of space for parallel processing
    from sklearn.cluster import KMeans

    encodings_list = [encodings]
    if random_scaling:
        n_clusters_list = [n_clusters]
        for i in range(n_partitions - 1):
            weight_vector = np.random.uniform(low=0.0, high=1.0, size=encodings.shape[1])
            encodings_list.append(np.multiply(encodings, weight_vector))
    else:
        n_clusters_list = [n_clusters] * n_partitions
    assert len(encodings_list) * len(n_clusters_list) == n_partitions
    if n_partitions != 1:
        n_init = 3
        init = 'k-means++'
    else:
        n_init = 10
        init = 'k-means++'

    print('Number of encodings: {}, number of n_clusters: {}, number of inits: '.format(len(encodings_list),
                                                                                        len(n_clusters_list)), n_init)

    kmeans_pre_labels_list = []
    kmeans_post_labels_list = []
    num_labels = len(np.unique(gt_labels))
    for n_clusters in tqdm(n_clusters_list, desc='get_partitions_kmeans_n_clusters'):
        for encodings in tqdm(encodings_list, desc='get_partitions_kmeans_encodings'):
            while True:
                kmeans_pre = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=3000).fit(encodings)
                kmeans_pre_labels = kmeans_pre.labels_.copy()
                kmeans_post = kmeans_post_process(kmeans_pre, semi_sup_labels)
                kmeans_post_labels = kmeans_post.labels_.copy()
                uniques, counts = np.unique(kmeans_post.labels_, return_counts=True)
                num_big_enough_clusters = np.sum(counts >= n_shot + n_query)
                if num_big_enough_clusters >= num_labels:
                    break
                else:
                    # if FLAGS.datasource == 'miniimagenet' and num_big_enough_clusters > 0.1 * n_clusters:
                    #     break
                    tqdm.write("Too few classes ({}) with greater than {} examples.".format(num_big_enough_clusters,
                                                                                            n_shot + n_query))
                    tqdm.write('Frequency: {}'.format(counts))

            kmeans_pre_labels_list.append(kmeans_pre_labels)
            kmeans_post_labels_list.append(kmeans_post_labels)

    partitions = []
    for labels in kmeans_pre_labels_list:
        partitions.append(Partition(labels=labels, n_way=n_way, n_shot=n_shot, n_query=n_query))

    partitions_merged = []
    for labels in kmeans_post_labels_list:
        partitions_merged.append(Partition(labels=labels, n_way=n_way, n_shot=n_shot, n_query=n_query))
    
    #visualize_partition(encodings, kmeans_pre_labels_list[0], kmeans_post_labels_list[0], gt_labels)
    measure_dup_sample_rate(partitions, gt_labels, n_way, n_shot)
    measure_dup_sample_rate(partitions_merged, gt_labels, n_way, n_shot)
    import pdb
    pdb.set_trace()
    return partitions_merged

def get_semi_sup_labels(labels):
    semi_sup_labels = labels.copy()
    num_unlabeled = int(0.9 * len(labels))
    unlabeled_idxs = np.random.choice(len(labels), num_unlabeled, replace=False)
    semi_sup_labels[unlabeled_idxs] = -1
    return semi_sup_labels

def load(opt, splits):
    encodings_dir = os.path.join(DATA_DIR, '{}_encodings'.format(opt['data.encoder']))
    filenames = os.listdir(encodings_dir)

    ret = { }
    for split in splits:
        if split in ['val', 'test'] and opt['data.test_way'] != 0:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['val', 'test'] and opt['data.test_shot'] != 0:
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot']

        if split in ['val', 'test'] and opt['data.test_query'] != 0:
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query']

        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
            mode = opt['data.test_mode']
        else:
            n_episodes = opt['data.train_episodes']
            mode = opt['data.train_mode']


        split_filename = [filename for filename in filenames if opt['data.dataset'] in filename and split in filename]
        assert len(split_filename) == 1
        split_filename = os.path.join(encodings_dir, split_filename[0])

        split_data = np.load(split_filename, allow_pickle=True).item()
        images = split_data['X']    # (index, H, W, C)
        labels = split_data['Y']
        encodings = split_data['Z']
        semi_sup_labels = get_semi_sup_labels(labels) # unlabeled points will have label -1

        if mode == 'ground_truth':
            if opt['data.dataset'] == 'celeba':
                annotations_filename = os.path.join(DATA_DIR, 'celeba/cropped/Anno/list_attr_celeba.txt')
                partitions = celeba_partitions(labels=labels, split=split, annotations_filename=annotations_filename, n_way=n_way, n_shot=n_support, n_query=n_query)
            else:
                partitions = [Partition(labels=labels, n_way=n_way, n_shot=n_support, n_query=n_query)]

        elif mode == 'kmeans':
            partitions = get_partitions_kmeans(
                encodings=encodings,
                semi_sup_labels=semi_sup_labels,
                gt_labels=labels,
                n_way=n_way, n_shot=n_support, n_query=n_query,
                n_partitions=opt['data.partitions'], n_clusters=opt['data.clusters']
            )

        elif mode == 'random':
            partitions = [Partition(labels=np.random.choice(opt['data.clusters'], size=labels.shape, replace=True), n_way=n_way, n_shot=n_support, n_query=n_query) for i in range(opt['data.partitions'])]
        else:
            raise ValueError
        ret[split] = TaskLoader(data=images, partitions=partitions, n_way=n_way, n_shot=n_support, n_query=n_query,
                                cuda=opt['data.cuda'], length=n_episodes)

    return ret
