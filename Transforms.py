import numpy as np
import torch
import random
import cv2
import numpy


class Scale(object):
    """Resize image and label to a fixed size"""

    def __init__(self, wi, he):
        self.w = wi
        self.h = he

    def __call__(self, img, label):
        img = cv2.resize(img, (self.w, self.h))
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return [img, label]


class Resize(object):
    """Resize with random short-side scaling and max-size constraint"""

    def __init__(self, min_size, max_size, strict=False):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.strict = strict

    def get_size(self, image_size):
        w, h = image_size

        if not self.strict:
            size = random.choice(self.min_size)

            if self.max_size is not None:
                min_org = float(min(w, h))
                max_org = float(max(w, h))
                if max_org / min_org * size > self.max_size:
                    size = int(round(self.max_size * min_org / max_org))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                return (int(size * h / w), size)
            else:
                return (size, int(size * w / h))
        else:
            if w < h:
                return (self.max_size, self.min_size[0])
            else:
                return (self.min_size[0], self.max_size)

    def __call__(self, image, label):
        size = self.get_size(image.shape[:2])
        image = cv2.resize(image, size)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return (image, label)


class RandomCropResize(object):
    """Random crop followed by resize back to original size"""

    def __init__(self, crop_area):
        self.cw = crop_area
        self.ch = crop_area

    def __call__(self, img, label):
        if random.random() < 0.5:
            h, w = img.shape[:2]
            x1 = random.randint(0, self.ch)
            y1 = random.randint(0, self.cw)

            img_crop = img[y1:h - y1, x1:w - x1]
            label_crop = label[y1:h - y1, x1:w - x1]

            img_crop = cv2.resize(img_crop, (w, h))
            label_crop = cv2.resize(label_crop, (w, h), interpolation=cv2.INTER_NEAREST)
            return img_crop, label_crop

        return [img, label]


class RandomFlip(object):
    """Random horizontal and vertical flip"""

    def __call__(self, image, label):
        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return [image, label]


class RandomExchange(object):
    """Randomly swap pre- and post-event images (for bi-temporal input)"""

    def __call__(self, image, label):
        if random.random() < 0.5:
            pre_img = image[:, :, 0:3]
            post_img = image[:, :, 3:6]
            image = numpy.concatenate((post_img, pre_img), axis=2)
        return [image, label]


class Normalize(object):
    """Normalize image using dataset mean and std"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        image = image.astype(np.float32) / 255.0
        label = np.ceil(label / 255)

        for i in range(6):
            image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]

        return [image, label]


class GaussianNoise(object):
    """Add Gaussian noise to image"""

    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, image, label):
        noise = np.random.normal(0, self.std, size=image.shape)
        image = image + noise.astype(np.float32)
        return [image, label]


class ToTensor(object):
    """Convert numpy arrays to PyTorch tensors"""

    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, image, label):
        if self.scale != 1:
            h, w = label.shape[:2]
            image = cv2.resize(image, (w, h))
            label = cv2.resize(
                label,
                (int(w / self.scale), int(h / self.scale)),
                interpolation=cv2.INTER_NEAREST
            )

        image = image[:, :, ::-1].copy()      # BGR → RGB
        image = image.transpose((2, 0, 1))    # HWC → CHW

        image_tensor = torch.from_numpy(image)
        label_tensor = torch.LongTensor(
            np.array(label, dtype=np.int_)
        ).unsqueeze(0)

        return [image_tensor, label_tensor]


class Compose(object):
    """Compose multiple transforms"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
