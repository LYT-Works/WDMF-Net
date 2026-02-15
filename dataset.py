import cv2
import numpy
import torch.utils.data
import os


class Dataset(torch.utils.data.Dataset):
    """Bi-temporal change detection dataset"""
    def __init__(self, dataset, file_root='data/', transform=None):
        self.file_list = open(file_root + '/' + dataset + '/list/' + dataset + '.txt').read().splitlines()
        self.pre_images = [file_root + '/' + dataset + '/A/' + x for x in self.file_list]
        self.post_images = [file_root + '/' + dataset + '/B/' + x for x in self.file_list]

        if not os.path.exists(file_root + '/' + dataset + '/label/'):
            self.gts = [file_root + '/label.png']
        else:
            self.gts = [file_root + '/' + dataset + '/label/' + x for x in self.file_list]
        self.transform = transform

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        """Return (H×W×6 image, H×W label)"""
        pre_image = cv2.imread(self.pre_images[idx])
        post_image = cv2.imread(self.post_images[idx])

        try:
            label = cv2.imread(self.gts[idx], 0)
        except IndexError:
            label = cv2.imread(self.gts[0], 0)

        img = numpy.concatenate((pre_image, post_image), axis=2)

        if self.transform:
            img, label = self.transform(img, label)

        return img, label

    def get_img_info(self, idx):
        """Return image spatial size"""
        img = cv2.imread(self.pre_images[idx])
        return {"height": img.shape[0], "width": img.shape[1]}
