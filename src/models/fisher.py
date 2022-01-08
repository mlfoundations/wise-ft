import copy
import itertools
import os

import torch
import torch.nn.functional as F

from tqdm.auto import tqdm

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.modeling import ImageClassifier
from src.models import utils
from src.models.utils import LabelSmoothing, fisher_save

import src.datasets as datasets

###############################################################################
# TODO: Make these args
#######################################
_TRAIN_PREPROCESSING = False
# Set this to a positive integer for faster testing.
_N_EXAMPLES_PER_EPOCH = None
###############################################################################


def compute_fisher(args):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    assert args.train_dataset is not None, "Please provide a training dataset."
    assert args.fisher is not None, "Please provide a path to save the Fisher to through --fisher."

    save_path, = args.fisher
    save_path = os.path.expanduser(save_path)

    # Copy the args so we can force the batch size to be 1 without affecting
    # other parts of the code base.
    args = copy.deepcopy(args)
    args.batch_size = 1

    model = ImageClassifier.load(os.path.expanduser(args.load))
    model.process_images = True

    if _TRAIN_PREPROCESSING:
        preprocess_fn = model.train_preprocess
    else:
        preprocess_fn = model.val_preprocess

    input_key = 'images'

    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        preprocess_fn,
        location=args.data_location,
        # TODO: See if this needs to be set to 1.
        batch_size=args.batch_size
    )

    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    # NOTE: Not sure if label smoothing makes sense for Fisher
    # computation.
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    # Initialize the Fisher accumulators.
    for p in model.parameters():
        p.grad2_acc = torch.zeros_like(p.data)
        p.grad_counter = 0

    for k in range(args.epochs):
        
        model.train()
        data_loader = get_dataloader(
            dataset, is_train=_TRAIN_PREPROCESSING, args=args, image_encoder=None)

        if _N_EXAMPLES_PER_EPOCH is not None:
            data_loader = itertools.islice(data_loader, 100)

        for i, batch in enumerate(tqdm(data_loader, leave=False, desc="Computing Fisher")):
            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].cuda()

            logits = utils.get_logits(inputs, model)

            target = torch.multinomial(F.softmax(logits, dim=-1), 1).detach().view(-1)
            loss = loss_fn(logits, target)

            model.zero_grad()
            loss.backward()

            for p in model.parameters():
                if p.grad is not None:
                    p.grad2_acc += p.grad.data ** 2
                    p.grad_counter += 1

    fisher = {}

    for name, p in model.named_parameters():
        if name.startswith('module.'):
            name = name[len('module.'):]
        if p.grad_counter == 0:
            print(f'No gradients found for parameter: {name}')
            del p.grad2_acc
        else:
            p.grad2_acc /= p.grad_counter
            fisher[name] = p.grad2_acc

    fisher_save(fisher, save_path)


if __name__ == '__main__':
    args = parse_arguments()
    compute_fisher(args)
