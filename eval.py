import argparse
import os
import sys 

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.cluster_and_log_utils import log_accs_from_preds
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups




parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
parser.add_argument('--use_ssb_splits', action='store_true', default=True)

parser.add_argument('--transform', type=str, default='imagenet')
parser.add_argument('--prompt_type', type=str, default='all')
parser.add_argument('--pretrained_model_path', type=str, required=True)
parser.add_argument('--model_type', type=str, default='dino', choices=['dino', 'clip'])


# ----------------------
# INIT
# ----------------------
args = parser.parse_args()
device = torch.device('cuda')
args = get_class_splits(args)

# Set up class splits
args.num_labeled_classes = len(args.train_classes)
args.num_unlabeled_classes = len(args.unlabeled_classes)
args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes


def test(model, test_loader, save_name, args):
    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=0, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


def load_model(args):
    # ----------------------
    # Hyper-parameters
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.patch_size = 16

    # ----------------------
    # BASE MODEL
    # ----------------------
    if args.model_type == 'dino':
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model = nn.Sequential(backbone, projector).cuda()

    print(f"Loading model from {args.pretrained_model_path}")
    state_dict = torch.load(args.pretrained_model_path, map_location="cpu")
    
    if 'model' in state_dict:
        model.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict)
    
    model.eval()
    return model


if __name__ == "__main__":
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')
    torch.backends.cudnn.benchmark = True

    # Load model
    model = load_model(args)

    # DATASETS
    _, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_dataset, test_dataset, unlabelled_train_examples_test, _ = get_datasets(
        args.dataset_name, None, test_transform, args
    )

    # DATALOADERS
    test_loader_unlabelled = DataLoader(
        unlabelled_train_examples_test, 
        num_workers=args.num_workers,
        batch_size=args.batch_size, 
        shuffle=False, 
        pin_memory=False
    )
    test_loader_labelled = DataLoader(
        test_dataset, 
        num_workers=args.num_workers,
        batch_size=args.batch_size, 
        shuffle=False, 
        pin_memory=False
    )
    
    # ----------------------
    # EVAL
    # ----------------------
    print("\nEvaluating on unlabelled train examples:")
    all_acc, old_acc, new_acc = test(model, test_loader_unlabelled, save_name='Eval ACC Unlabelled', args=args)
    print('Unlabelled Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
