## Instructions for downloading datasets

### ImageNet distribution shifts

```bash
export DATA_LOCATION=~/data
cd $DATA_LOCATION
```

#### [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)


```bash
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
tar -xvzf imagenet-a.tar
rm imagenet-a.tar
```

#### [ImageNet-R](https://github.com/hendrycks/imagenet-r)

```bash
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
tar -xvzf imagenet-r.tar
rm imagenet-r.tar
```

#### [ImageNet Sketch](https://github.com/HaohanWang/ImageNet-Sketch)

Download links:
- from [Google Drive](https://drive.google.com/open?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA)
- from [Kaggle](https://www.kaggle.com/wanghaohan/imagenetsketch)

#### [ImageNet V2](https://github.com/modestyachts/ImageNetV2)

```bash
wget https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz
tar -xvf imagenetv2-matched-frequency.tar.gz
rm imagenetv2-matched-frequency.tar.gz
```

#### [ObjectNet](https://objectnet.dev/)

```bash
wget https://objectnet.dev/downloads/objectnet-1.0.zip
unzip objectnet-1.0.zip
rm objectnet-1.0.zip
```

#### [YTBB Robust and ImageNet Vid-Robust](https://modestyachts.github.io/natural-perturbations-website/)

```bash
wget https://do-imagenet-classifiers-generalize-across-time.s3-us-west-2.amazonaws.com/imagenet_vid_ytbb_robust.tar.gz
tar -xvf imagenet_vid_ytbb_robust.tar.gz
mv imagenet_vid_ytbb_robust/* .
rm -rf imagenet_vid_ytbb_robust*
```

### WILDS distribution shifts

```bash
export DATA_LOCATION=~/data
python utils/download_wilds_datasets.py
```


### CIFAR distribution shifts

```bash
export DATA_LOCATION=~/data

# CIFAR10.1
mkdir -p $DATA_LOCATION/CIFAR-10.1
wget https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy -P $DATA_LOCATION/CIFAR-10.1
wget https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy -P $DATA_LOCATION/CIFAR-10.1

# CIFAR10.2
mkdir -p $DATA_LOCATION/CIFAR-10.2
wget https://github.com/modestyachts/cifar-10.2/raw/61b0e3ac09809a2351379fb54331668cc9c975c4/cifar102_test.npy -P $DATA_LOCATION/CIFAR-10.2
wget https://github.com/modestyachts/cifar-10.2/raw/61b0e3ac09809a2351379fb54331668cc9c975c4/cifar102_train.npy -P $DATA_LOCATION/CIFAR-10.2
```
