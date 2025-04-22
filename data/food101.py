import numpy as np
from copy import deepcopy
from torchvision.datasets import Food101
from data.data_utils import subsample_instances
from config import food101_root

class CustomFood101(Food101):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        uq_idx = self.uq_idxs[idx]
        return img, label, uq_idx

def subsample_dataset(dataset, idxs):
    if len(idxs) == 0: return None
    dataset._labels = [dataset._labels[i] for i in idxs]
    dataset._image_files = [dataset._image_files[i] for i in idxs]
    dataset.uq_idxs = dataset.uq_idxs[idxs]
    return dataset


def subsample_classes(dataset, include_classes):
    cls_idxs = [x for x, l in enumerate(dataset._labels) if l in include_classes]
    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: include_classes.index(x)
    return dataset

def get_food101_datasets(train_transform, test_transform, 
                         train_classes=range(80), prop_train_labels=0.8,
                         split_train_val=False, seed=0):
    np.random.seed(seed)

    train_dataset = CustomFood101(
        root=food101_root,
        split='train',
        transform = train_transform,
        download=True
    )
    train_dataset_labelled = subsample_classes(deepcopy(train_dataset), train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    unlabelled_idxs = list(set(train_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs))
    train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset), unlabelled_idxs)

    test_dataset = CustomFood101(
        root=food101_root,
        split='test',
        transform=test_transform,
    )

    all_classes = list(train_classes) + list(set(range(101)) - set(train_classes))
    target_xform = lambda x: all_classes.index(x)
    test_dataset.target_transform = target_xform
    train_dataset_unlabelled.target_transform = target_xform

    return{
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'test': test_dataset
    }
