import os

import numpy as np

import torch

from src.models.eval import evaluate
from src.models.finetune import finetune
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments

def wise_ft(args):
    assert args.save is not None, 'Please provide a path to store models'
    
    if args.load is None:
        # Build and save zero-shot model
        image_encoder = ImageEncoder(args, keep_lang=True)
        classification_head = get_zeroshot_classifier(args, image_encoder.model)
        delattr(image_encoder.model, 'transformer')
        classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
        zeroshot_checkpoint = os.path.join(args.save, 'zeroshot.pt')
        classifier.save(zeroshot_checkpoint)

        # Standard fine-tuning
        args.load = zeroshot_checkpoint
        args.save = os.path.join(args.save, 'finetuned')
        finetuned_checkpoint = finetune(args)
    else:
        # No need to compute things from stratch
        assert len(args.load) == 2
        zeroshot_checkpoint, finetuned_checkpoint = args.load

    # Load models
    zeroshot = ImageClassifier.load(zeroshot_checkpoint)
    finetuned = ImageClassifier.load(finetuned_checkpoint)
    theta_0 = {k: v.clone() for k, v in zeroshot.state_dict().items()}
    theta_1 = {k: v.clone() for k, v in finetuned.state_dict().items()}
    del zeroshot

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    alphas = args.alpha
    for alpha in alphas:
        args.alpha = alpha

        # interpolate between all weights in the checkpoints
        theta = {
            key: (1-alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0.keys()
        }

        # update the model (in-place) acccording to the new weights
        finetuned.load_state_dict(theta)

        # save model
        finetuned.save(os.path.join(args.save, f'wise_ft_alpha={alpha:.3f}.pt'))

        # evaluate
        evaluate(finetuned, args)


if __name__ == '__main__':
    args = parse_arguments()
    wise_ft(args)



