#!/usr/bin/env bash
device=cpu
dataset=miniimagenet
date=20181003
way=5
shot=1
query=15
encoder=swav
clusters=100
partitions=20
hdim=64
train_mode=ground_truth
python scripts/train/few_shot/run_train.py \
    --data.dataset ${dataset} \
    --data.way ${way} \
    --data.shot ${shot} \
    --data.query ${query} \
    --data.test_way 5 \
    --data.test_shot 1 \
    --data.test_query 5 \
    --data.train_episodes 100 \
    --data.test_episodes 100 \
    --data.encoder ${encoder} \
    --data.train_mode ${train_mode} \
    --data.test_mode ground_truth \
    --data.clusters ${clusters} \
    --data.partitions ${partitions} \
    --model.x_dim 3,84,84 \
    --model.hid_dim ${hdim} \
    --train.epochs 600 \
    --log.exp_dir log \
    --log.date ${date}
train_mode=kmeans