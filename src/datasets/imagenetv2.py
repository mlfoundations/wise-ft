from PIL import Image

from imagenetv2_pytorch import ImageNetV2Dataset

from .imagenet import ImageNet

class ImageNetV2DatasetWithPaths(ImageNetV2Dataset):
    def __getitem__(self, i):
        img, label = Image.open(self.fnames[i]), int(self.fnames[i].parent.name)
        if self.transform is not None:
            img = self.transform(img)
        return {
            'images': img,
            'labels': label,
            'image_paths': str(self.fnames[i])
        }

class ImageNetV2(ImageNet):
    def get_test_dataset(self):
        return ImageNetV2DatasetWithPaths(transform=self.preprocess, location=self.location)
