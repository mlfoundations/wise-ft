import os
from .imagenet import ImageNet


class ImageNetSketch(ImageNet):

    def populate_train(self):
        pass

    def get_test_path(self):
        return os.path.join(self.location, 'sketch')
