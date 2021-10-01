import os
import json
from pathlib import Path
import PIL

import numpy as np

import torch
from torchvision import datasets
from torchvision.transforms import Compose

from .common import ImageFolderWithPaths, SubsetSampler
from .imagenet import ImageNet, ImageNetSubsampleValClasses


def get_metadata():
    metadata = Path(__file__).parent / 'objectnet_metadata'

    with open(metadata / 'folder_to_objectnet_label.json', 'r') as f:
        folder_map = json.load(f)
        folder_map = {v: k for k, v in folder_map.items()}
    with open(metadata / 'objectnet_to_imagenet_1k.json', 'r') as f:
        objectnet_map = json.load(f)

    with open(metadata / 'pytorch_to_imagenet_2012_id.json', 'r') as f:
        pytorch_map = json.load(f)
        pytorch_map = {v: k for k, v in pytorch_map.items()}

    with open(metadata / 'imagenet_to_label_2012_v2', 'r') as f:
        imagenet_map = {v.strip(): str(pytorch_map[i]) for i, v in enumerate(f)}

    folder_to_ids, class_sublist = {}, []
    classnames = []
    for objectnet_name, imagenet_names in objectnet_map.items():
        imagenet_names = imagenet_names.split('; ')
        imagenet_ids = [int(imagenet_map[imagenet_name]) for imagenet_name in imagenet_names]
        class_sublist.extend(imagenet_ids)
        folder_to_ids[folder_map[objectnet_name]] = imagenet_ids

    class_sublist = sorted(class_sublist)
    class_sublist_mask = [(i in class_sublist) for i in range(1000)]
    classname_map = {v: k for k, v in folder_map.items()}
    return class_sublist, class_sublist_mask, folder_to_ids, classname_map


def crop(img):
    width, height = img.size
    cropArea = (2, 2, width - 2, height - 2)
    img = img.crop(cropArea)
    return img


class ObjectNetDataset(datasets.ImageFolder):

    def __init__(self, label_map, path, transform):
        self.label_map = label_map
        super().__init__(path, transform=transform)
        self.samples = [
            d for d in self.samples
            if os.path.basename(os.path.dirname(d[0])) in self.label_map
        ]
        self.imgs = self.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        label = os.path.basename(os.path.dirname(path))
        return {
            'images': sample,
            'labels': self.label_map[label],
            'image_paths': path
        } 


class ObjectNetBase(ImageNet):
    def __init__(self, *args, **kwargs):
        (self._class_sublist,
         self.class_sublist_mask,
         self.folders_to_ids,
         self.classname_map) = get_metadata()
        
        super().__init__(*args, **kwargs)

        self.classnames = sorted(list(self.folders_to_ids.keys()))
        self.rev_class_idx_map = {}
        self.class_idx_map = {}
        for idx, name in enumerate(self.classnames):
            self.rev_class_idx_map[idx] = self.folders_to_ids[name]
            for imagenet_idx in self.rev_class_idx_map[idx]:
                self.class_idx_map[imagenet_idx] = idx

        self.crop = crop
        self.preprocess = Compose([crop, self.preprocess])
        self.classnames = [self.classname_map[c].lower() for c in self.classnames]

    def populate_train(self):
        pass

    def get_test_dataset(self):
        subdir = 'objectnet-1.0/images'
        valdir = os.path.join(self.location, subdir)
        label_map = {name: idx for idx, name in enumerate(sorted(list(self.folders_to_ids.keys())))}
        return ObjectNetDataset(label_map, valdir, transform=self.preprocess)

    def project_logits(self, logits, device):
        if isinstance(logits, list) or isinstance(logits, tuple):
            return [self.project_logits(l, device) for l in logits]
        if logits.shape[1] == 113:
            return logits
        if torch.is_tensor(logits):
            logits = logits.cpu().numpy()
        logits_projected = np.zeros((logits.shape[0], 113))
        for k, v in self.rev_class_idx_map.items():
            logits_projected[:, k] = np.max(logits[:, v], axis=1).squeeze()
        return torch.tensor(logits_projected).to(device)

    def scatter_weights(self, weights):
        if weights.size(1) == 1000:
            return weights
        new_weights = torch.ones((weights.size(0), 1000)).to(weights.device) * -10e8
        for k, v in self.rev_class_idx_map.items():
            for vv in v:
                new_weights[:, vv] = weights[:, k]
        return new_weights



def accuracy(logits, targets, img_paths, args):
    assert logits.shape[1] == 113
    preds = logits.argmax(dim=1)
    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    return np.sum(preds == targets), len(preds)


class ObjectNetValClasses(ObjectNetBase):

    def get_test_sampler(self):
        idx_subsample_list = [range(x * 50, (x + 1) * 50) for x in self._class_sublist]
        idx_subsample_list = sorted([item for sublist in idx_subsample_list for item in sublist])
        
        sampler = SubsetSampler(idx_subsample_list)
        return sampler

    def get_test_dataset(self):
        return ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)

    def project_labels(self, labels, device):
        projected_labels =  [self.class_idx_map[int(label)] for label in labels]
        return torch.LongTensor(projected_labels).to(device)


class ObjectNet(ObjectNetBase):

    def accuracy(self, logits, targets, img_paths, args):
        return accuracy(logits, targets, img_paths, args)
