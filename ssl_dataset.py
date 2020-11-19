import torch

from utils import split_ssl_data
from randaugment import RandAugment

from PIL import Image
import numpy as np
import copy

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import PIL
import random

mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]


def get_transform(mean, std, transform_list, train=True):
    if train:
        trs = transforms.Compose([transform_list[key] for key in transform_list])
        print("Weak augmentation: ")
        print(trs)
        return trs
    else:
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])


def construct_transforms(mean, std, args):
    # flip and crop are on by default
    transform_list = {'flip': transforms.RandomHorizontalFlip(), 'crop': transforms.RandomCrop(32, padding=4)}

    # parse arguments

    if args.translate:
        transform_list['translate_x'] = Translate_x()
        transform_list['translate_y'] = Translate_y()

    if args.contrast:
        transform_list['contrast'] = AddContrast()

    # tensor and normalize are on by default
    transform_list['to_tensor'] = transforms.ToTensor()
    transform_list['normalize'] = transforms.Normalize(mean, std)

    # noise addition happens to a tensor not an image, hence must be last
    if args.noise:
        transform_list['noise'] = AddGaussianNoise(std=0.03)

    return transform_list


class Translate_x(object):
    """
    Performs affine translation of the IMG object along the x-axis
    """

    def __init__(self, magnitude=0.125):
        v = random.uniform(0, magnitude)
        v = v * random.choice([-1, 1])
        self.v = v

    def __call__(self, img):
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, self.v * img.size[0], 0, 1, 0))

    def __repr__(self):
        return self.__class__.__name__ + '(v={})'.format(self.v)


class Translate_y(object):
    """
    Performs affine translation of the IMG object along the y-axis
    """

    def __init__(self, magnitude=0.125):
        v = random.uniform(0, magnitude)
        v = v * random.choice([-1, 1])
        self.v = v

    def __call__(self, img):
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, self.v * img.size[1]))

    def __repr__(self):
        return self.__class__.__name__ + '(v={})'.format(self.v)


class AddGaussianNoise(object):
    """
    Adds gaussian noise to the tensor of the image
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddContrast(object):
    def __init__(self, min_val=0.65, max_val=1.0):
        v = random.uniform(min_val, max_val)
        self.v = v

    def __call__(self, img):
        return ImageEnhance.Contrast(img).enhance(self.v)

    def __repr__(self):
        return self.__class__.__name__ + '(v={})'.format(self.v)


class SSL_Dataset:
    """
    SSL_Dataset class gets cifar10 from torchvision.datasets,
    separates labeled and unlabeled data,
    and returns BasicDataset: torch.utils.data.Dataset
    """

    def __init__(self,
                 name='cifar10',
                 train=True,
                 num_classes=10,
                 data_dir='./data',
                 args=None):
        """
        Args
            name: name of dataset in torchvision.datasets  = cifar.10
            train: True means the dataset is training dataset (default=True)
            num_classes: number of label classes
            data_dir: path of directory, where data is downloaed or stored.
        """

        self.name = name
        self.train = train
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.transform_list = construct_transforms(mean[name], std[name], args)
        self.transform = get_transform(mean[name], std[name], self.transform_list, train)

    def get_data(self):
        """
        get_data returns data (images) and targets (labels) from CIFAR-10
        """
        dset = getattr(torchvision.datasets, self.name.upper())
        dset = dset(self.data_dir, train=self.train, download=True)
        data, targets = dset.data, dset.targets
        return data, targets

    def get_dset(self, use_strong_transform=False,
                 strong_transform=None, onehot=False):
        """
        get_dset returns class BasicDataset, containing the returns of get_data.
        
        Args
            use_strong_tranform: If True, returned dataset generates a pair of weak and strong augmented images.
            strong_transform: list of strong_transform (augmentation) if use_strong_transform is True
            onehot: If True, the label is not integer, but one-hot vector.
        """

        data, targets = self.get_data()
        num_classes = self.num_classes
        transform = self.transform
        data_dir = self.data_dir

        return BasicDataset(data, targets, num_classes, transform,
                            use_strong_transform, strong_transform, onehot)

    # For test time transformation
    def get_dset_clean(self, use_strong_transform=False,
                       strong_transform=None, onehot=False):
        """
        get_dset returns class BasicDataset, containing the returns of get_data.

        Args
            use_strong_tranform: If True, returned dataset generates a pair of weak and strong augmented images.
            strong_transform: list of strong_transform (augmentation) if use_strong_transform is True
            onehot: If True, the label is not integer, but one-hot vector.
        """

        data, targets = self.get_data()
        num_classes = self.num_classes
        transform = None
        data_dir = self.data_dir

        return BasicDataset(data, targets, num_classes, transform,
                            use_strong_transform, strong_transform, onehot)

    def get_ssl_dset(self, num_labels, index=None, include_lb_to_ulb=True,
                     use_strong_transform=True, strong_transform=None,
                     onehot=False):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.
        
        Args:
            num_labels: number of labeled data.
            index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
            include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
            use_strong_transform: If True, unlabeld dataset returns weak & strong augmented image pair. 
                                  If False, unlabeled datasets returns only weak augmented image.
            strong_transform: list of strong transform (RandAugment in FixMatch)
            oenhot: If True, the target is converted into onehot vector.
            
        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeld data)
        """

        data, targets = self.get_data()
        num_classes = self.num_classes
        transform = self.transform
        data_dir = self.data_dir

        lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(data, targets,
                                                                    num_labels, num_classes,
                                                                    index, include_lb_to_ulb)
        lb_dset = BasicDataset(lb_data, lb_targets, num_classes,
                               transform, False, None, onehot)

        ulb_dset = BasicDataset(data, targets, num_classes,
                                transform, use_strong_transform, strong_transform, onehot)

        return lb_dset, ulb_dset


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 use_strong_transform=False,
                 strong_transform=None,
                 *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
        """
        super(BasicDataset, self).__init__()
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.use_strong_transform = use_strong_transform

        self.transform = transform
        if use_strong_transform:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugment(3, 5))
        else:
            self.strong_transform = strong_transform

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """

        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_
        # set augmented images

        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)
            if not self.use_strong_transform:
                return img_w, target
            else:
                return img_w, self.strong_transform(img), target

    def __len__(self):
        return len(self.data)
