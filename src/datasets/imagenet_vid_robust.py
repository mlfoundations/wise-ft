import os
import torch
import torchvision.datasets as datasets

from .imagenet import ImageNet
import numpy as np
import pathlib
import json

from .common import ImageFolderWithPaths, SubsetSampler


class VidRobustDataset(ImageFolderWithPaths):
    def __init__(self, label_map, path, transform):
        self.label_map = label_map
        super().__init__(path, transform=transform)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        label_key = '/'.join(data['image_paths'].split('/')[-3:])
        data['labels'] = self.label_map[label_key][0]
        return data

class ImageNetVidRobustBase(ImageNet):
    def __init__(self, *args, **kwargs):
        data_loc = pathlib.Path(kwargs.get('location', '~')) / 'imagenet_vid_ytbb_robust/imagenet-vid-robust'
        with open((data_loc / 'misc/wnid_map.json').resolve()) as f:
            self.wnid_map = json.load(f)
        with open((data_loc / 'misc/rev_wnid_map.json').resolve()) as f:
            self.rev_wnid_map = json.load(f)
        with open((data_loc / 'misc/imagenet_class_index.json').resolve()) as f:
            self.imagenet_class_index = json.load(f)
        with open((data_loc / 'misc/imagenet_vid_class_index.json').resolve()) as f:
            self.imagenet_vid_class_index = json.load(f)
        with open((data_loc / 'metadata/labels.json').resolve()) as f:
            self.label_map = json.load(f)
        with open((data_loc / 'metadata/pmsets.json').resolve()) as f:
            self.pmsets = json.load(f)

        rev_imagenet = {v[0] : k for k, v in self.imagenet_class_index.items()}
        rev_vid = {v[0] : k for k,v in self.imagenet_vid_class_index.items()}
        self.CLASS_IDX_LIST = sorted([int(rev_imagenet[k]) for k in self.wnid_map])
        self.CLASS_IDX_MAP = {int(rev_imagenet[k]) : int(rev_vid[v]) for k, v in self.wnid_map.items()}
        self.rev_class_idx_map = {int(rev_vid[k]): [int(rev_imagenet[elt]) for elt in v] for k, v in self.rev_wnid_map.items()}
        self.merge_op = 'max'

        super().__init__(*args, **kwargs)

        self.classnames = [self.imagenet_vid_class_index[str(i)][1] for i in range(30)]

    def populate_train(self):
        pass

    def project_logits(self, logits, device):
        if isinstance(logits, list) or isinstance(logits, tuple):
            return [self.project_logits(l, device) for l in logits]
        if logits.shape[1] == 30:
            return logits
        if torch.is_tensor(logits):
            logits = logits.cpu().numpy()
        logits_projected = np.zeros((logits.shape[0], 30))
        for k, v in self.rev_class_idx_map.items():
            if self.merge_op == 'mean':
                logits_projected[:, k] = np.mean(logits[:, v], axis=1).squeeze()
            elif self.merge_op == 'median':
                logits_projected[:, k] = np.median(logits[:, v], axis=1).squeeze()
            elif self.merge_op == 'max':
                logits_projected[:, k] = np.max(logits[:, v], axis=1).squeeze()
            elif self.merge_op == 'sum':
                logits_projected[:, k] = np.sum(logits[:, v], axis=1)
            else:
                raise Exception(f'unsupported merge operation {merge_op} not allowed')
        return torch.tensor(logits_projected).to(device)

    def scatter_weights(self, weights):
        if weights.size(1) == 1000:
            return weights
        new_weights = torch.ones((weights.size(0), 1000)).to(weights.device) * -10e10
        for k, v in self.rev_class_idx_map.items():
            for vv in v:
                new_weights[:, vv] = weights[:, k]
        return new_weights


class ImageNetVidRobustValClasses(ImageNetVidRobustBase):

    def post_loop_metrics(self, targets, logits, image_paths, args):
        logits = logits.numpy()
        targets = targets.numpy()
        return {'acc' : self.score_predictions(logits, targets)}

    def score_predictions(self, logits_projected, targets):
        preds = logits_projected.argmax(axis=1)
        acc = np.sum(np.equal(preds, targets))
        n = len(preds)
        return acc/n

    def get_test_sampler(self):
        idx_subsample_list = [range(x * 50, (x + 1) * 50) for x in self.CLASS_IDX_LIST]
        idx_subsample_list = sorted([item for sublist in idx_subsample_list for item in sublist])

        sampler = SubsetSampler(idx_subsample_list)
        return sampler

    def project_labels(self, labels, device):
        labels = labels.cpu().numpy()
        labels_projected = torch.tensor([self.CLASS_IDX_MAP[label] for label in labels]).to(device)
        return labels_projected


class ImageNetVidRobust(ImageNetVidRobustBase):

    def score_predictions(self, preds, pmsets):
        correct_anchor = 0
        correct_pmk = 0
        N = len(pmsets)
        wrong_map = {}
        for anchor, pmset in pmsets.items():
            pmset_correct = 0
            wrongs = []
            for elem in pmset:
                if np.argmax(preds[elem]) in self.label_map[elem]:
                    pmset_correct += 1
                else:
                    wrongs.append(elem)

            if np.argmax(preds[anchor]) in self.label_map[anchor]:
                correct_anchor  += 1
                pmset_correct += 1
                if len(wrongs) > 0:
                    wrong_map[anchor] = wrongs[-1]

            if pmset_correct == len(pmset) + 1:
                correct_pmk += 1

        return correct_anchor/N, correct_pmk/N

    def post_loop_metrics(self, labels, logits, image_paths, args):
        logits = logits.numpy()
        labels = labels.numpy()

        preds_dict = {}
        for i, img_name in enumerate(image_paths):
            preds_dict['val/' + img_name.split('val/')[1]] = logits[i]

        benign,pmk = self.score_predictions(preds_dict, self.pmsets)
        metrics_dict = {}
        metrics_dict['pm0'] = benign
        metrics_dict['pm10'] = pmk
        metrics_dict['merge_op'] = self.merge_op
        return metrics_dict

    def get_test_dataset(self):
        valdir = os.path.join(self.location, 'imagenet_vid_ytbb_robust/imagenet-vid-robust/val')
        return VidRobustDataset(self.label_map, valdir, transform=self.preprocess)

