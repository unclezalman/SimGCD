import os
import torchvision
import numpy as np
from copy import deepcopy

from data.data_utils import subsample_instances
from config import food101_root

class Food101Dataset(torchvision.datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        uq_idx = self.uq_idxs[idx]
        return img, label, uq_idx

def subsample_dataset(dataset, idxs):
    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True
    
    dataset.samples = np.array(dataset.samples)[mask].tolist()
    dataset.targets = np.array(dataset.targets)[mask].tolist()
    dataset.uq_idxs = dataset.uq_idxs[mask]
    
    dataset.samples = [[x[0], int(x[1])] for x in dataset.samples]
    dataset.targets = [int(x) for x in dataset.targets]
    return dataset

def subsample_classes(dataset, include_classes):
    cls_idxs = [x for x, l in enumerate(dataset.targets) if l in include_classes]
    dataset = subsample_dataset(dataset, cls_idxs)
    return dataset

def get_train_val_indices(train_dataset, val_split=0.2):
    train_classes = np.unique(train_dataset.targets)
    train_idxs, val_idxs = [], []
    
    for cls in train_classes:
        cls_idxs = np.where(train_dataset.targets == cls)[0]
        v_ = np.random.choice(cls_idxs, size=int(val_split*len(cls_idxs)), replace=False)
        t_ = [x for x in cls_idxs if x not in v_]
        train_idxs.extend(t_)
        val_idxs.extend(v_)
    return train_idxs, val_idxs

def get_food101_datasets(train_transform, test_transform, train_classes=range(50),
                         prop_train_labels=0.8, split_train_val=False, seed=0):
    np.random.seed(seed)
    

    train_dataset = Food101Dataset(root=os.path.join(food101_root, 'train'), 
                                  transform=train_transform)
    

    train_dataset_labelled = subsample_classes(deepcopy(train_dataset), train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)


    if split_train_val:
        train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
        train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
        val_dataset_labelled_split.transform = test_transform
    else:
        train_dataset_labelled_split, val_dataset_labelled_split = None, None


    unlabelled_indices = set(train_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset), list(unlabelled_indices))


    test_dataset = Food101Dataset(root=os.path.join(food101_root, 'test'), 
                                transform=test_transform)


    all_classes = list(train_classes) + list(set(test_dataset.targets) - set(train_classes))
    target_xform = {cls:i for i, cls in enumerate(all_classes)}
    
    test_dataset.target_transform = lambda x: target_xform[x]
    train_dataset_unlabelled.target_transform = lambda x: target_xform[x]

    return {
        'train_labelled': train_dataset_labelled_split if split_train_val else train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled_split if split_train_val else None,
        'test': test_dataset
    }

if __name__ == '__main__':
    datasets = get_food101_datasets(None, None)
    print({k: len(v) if v else 0 for k, v in datasets.items()})
