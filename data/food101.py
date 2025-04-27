import os
import torchvision
import numpy as np
from copy import deepcopy

from data.data_utils import subsample_instances
from config import food101_root
from torchvision.datasets import Food101

class CustomFood101(Food101):
    def __init__(self, *args, **kwargs):
        split = kwargs.pop('split', 'train')
        super().__init__(*args, split=split, **kwargs)
        self.uq_idxs = np.array(range(len(self)))
        
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        return img, target, self.uq_idxs[idx]

def subsample_dataset(dataset, idxs):
    if len(idxs) == 0:
        return None
        
    dataset.data = [dataset.data[i] for i in idxs]
    dataset._labels = [dataset._labels[i] for i in idxs]
    dataset.uq_idxs = dataset.uq_idxs[idxs]
    return dataset

def subsample_classes(dataset, include_classes):
    cls_idxs = [i for i, label in enumerate(dataset._labels) if label in include_classes]
    return subsample_dataset(dataset, cls_idxs)


def get_food101_datasets(train_transform, test_transform, train_classes=range(50),
                        prop_train_labels=0.8, split_train_val=False, seed=0):
    np.random.seed(seed)

    whole_train = CustomFood101(root=food101_root, split='train', 
                              transform=train_transform, download=True)
    test_dataset = CustomFood101(root=food101_root, split='test',
                               transform=test_transform, download=True)


    train_dataset_labelled = subsample_classes(deepcopy(whole_train), train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)


    labelled_idxs = set(train_dataset_labelled.uq_idxs)
    unlabelled_idxs = [i for i in range(len(whole_train)) if i not in labelled_idxs]
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_train), unlabelled_idxs)


    all_classes = list(train_classes) + list(set(range(101)) - set(train_classes))
    target_xform = {cls:i for i, cls in enumerate(all_classes)}
    
    for dataset in [train_dataset_labelled, train_dataset_unlabelled, test_dataset]:
        if dataset:
            dataset.target_transform = lambda x: target_xform[x]

    return {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'test': test_dataset,
        'val': None  # No separate val set
    }
