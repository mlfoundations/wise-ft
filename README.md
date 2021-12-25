# Robust fine-tuning of zero-shot models

This repository contains code for the paper [Robust fine-tuning of zero-shot models](https://arxiv.org/abs/2109.01903) by Mitchell Wortsman*, Gabriel Ilharco*,  Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo-Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, Ludwig Schmidt.

TLDR: We fine-tune zero-shot models while preserving or improving OOD accuracy at no extra computational cost during fine-tuning or inference.

### Abstract
*Large pre-trained models such as CLIP or ALIGN offer consistent accuracy across a range of data distributions when performing zero-shot inference (i.e., without fine-tuning on a specific dataset). Although existing fine-tuning approaches substantially improve accuracy in-distribution, they often reduce out-of-distribution robustness. We address this tension by introducing a simple and effective method for improving robustness:  ensembling the weights of the zero-shot and fine-tuned models (WiSE-FT). Compared to standard fine-tuning, WiSE-FT provides large accuracy improvements out-of-distribution, while  preserving high in-distribution accuracy. On ImageNet (in-distribution) and five derived distribution shifts, WiSE-FT improves out-of-distribution accuracy by 4 to 6 percentage points (pp) over prior work while increasing in-distribution accuracy by 1.6 pp. WiSE-FT achieves similarly large robustness improvements (2 to 23 pp) on a diverse set of six further distribution shifts, and in-distribution accuracy gains of 0.8 to 3.3 pp compared to standard fine-tuning on seven commonly used transfer learning datasets. These improvements come at no additional computational cost during fine-tuning or inference.*

### Summary figure

<p align="center">
<img src="images/figure1.png" alt="figure1"/>
</p>

## Code

### Overview

WiSE-FT can be implemented in a few lines of code in addition to standard fine-tuning, as shown below. See [src/wise_ft.py](src/wise_ft.py) for more details.

```python
# Load models
zeroshot = ImageClassifier.load(zeroshot_checkpoint)
finetuned = ImageClassifier.load(finetuned_checkpoint)
theta_0 = zeroshot.state_dict()
theta_1 = finetuned.state_dict()

# make sure checkpoints are compatible
assert set(theta_0.keys()) == set(theta_1.keys())

# interpolate between checkpoints with mixing coefficient alpha
theta = {
    key: (1-alpha) * theta_0[key] + alpha * theta_1[key]
    for key in theta_0.keys()
}

# update the model acccording to the new weights
finetuned.load_state_dict(theta)

# evaluate
evaluate(finetuned, args)
```

### Install dependencies

```bash
conda env create
conda activate wiseft
```

### Add directory to PYTHONPATH:

```bash
cd wise-ft
export PYTHONPATH="$PYTHONPATH:$PWD"
```

### Download data

When necessary, please refer to [datasets.md](datasets.md) for instructions on how to download datasets.

### Run WiSE-FT

Sample command when zeroshot and fine-tuned models are available:

```
python src/wise_ft.py   \
    --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch  \
    --load=models/zeroshot.pt,models/finetuned.pt  \
    --results-db=results.jsonl  \
    --save=models/wiseft  \
    --data-location=~/data \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
```

Sample command for running WiSE-FT from scratch using ViT-B/32:

```
python src/wise_ft.py   \
    --train-dataset=ImageNet  \
    --epochs=10  \
    --lr=0.00003  \
    --batch-size=512  \
    --cache-dir=cache  \
    --model=ViT-B/32  \
    --eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch  \
    --template=openai_imagenet_template  \
    --results-db=results.jsonl  \
    --save=models/wiseft/ViTB32  \
    --data-location=~/data \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
```

Note: the flag `--freeze-encoder` controls whether only a linear classifier is fine-tuned, or if all weights are fine-tuned (end-to-end).

### Plotting results

Sample command for generating a scatter plot:

```
python src/scatter_plot.py  \
    --eval-datasets=ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch  \
    --results-db=results.jsonl  \
    --save plots
```

We show samples of expected behavior below when running the commands above using ViT-B/16 (models can be downloaded  [here](https://drive.google.com/drive/folders/1f56kjpRKPiNSaUxNDtETEDRkbDkZnpCQ?usp=sharing)):

<p align="center">
  <img alt="ImageNet-Sketch" src="images/b16/ImageNetSketch.png" width="45%">
    &nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="ImageNet-A" src="images/b16/ImageNetA.png" width="45%">
</p>
<p align="center">
  <img alt="ImageNet-R" src="images/b16/ImageNetR.png" width="45%">
    &nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="ImageNetV2" src="images/b16/ImageNetV2.png" width="45%">
</p>
<p align="center">
  <img alt="ObjectNet" src="images/b16/ObjectNet.png" width="45%">
</p>

## Citing

If you found this repository useful, please consider citing:
```bibtex
@article{wortsman2021robust,
  title={Robust fine-tuning of zero-shot models},
  author={Wortsman, Mitchell and Ilharco, Gabriel and Kim, Jong Wook and Li, Mike and Kornblith, Simon and Roelofs, Rebecca and Gontijo-Lopes, Raphael and Hajishirzi, Hannaneh and Farhadi, Ali and Namkoong, Hongseok and Schmidt, Ludwig},
  journal={arXiv preprint arXiv:2109.01903},
  note={\url{https://arxiv.org/abs/2109.01903}},
  year={2021}
}
```
