# -*- coding: utf-8 -*-
""" The code in this script is adapted from `https://github.com/ajbrock/BigGAN-PyTorch`, more precisely:
        `https://github.com/ajbrock/BigGAN-PyTorch/blob/master/datasets.py`
    and is used for loading only the ImageNet dataset.
"""
import os
import os.path
import sys
from PIL import Image
import numpy as np
from tqdm import tqdm, trange

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url, check_integrity
import torch.utils.data as data
from torch.utils.data import DataLoader

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in tqdm(sorted(os.listdir(dir))):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dogball/xxx.png
        root/dogball/xxy.png
        root/dogball/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, load_in_mem=False,
                 index_filename='imagenet_imgs.npz', **kwargs):
        classes, class_to_idx = find_classes(root)
        # Load pre-computed image directory walk
        if os.path.exists(index_filename):
            print('Loading pre-saved Index file %s...' % index_filename)
            imgs = np.load(index_filename)['imgs']
        # If first time, walk the folder directory and save the
        # results to a pre-computed file.
        else:
            print('Generating  Index file %s...' % index_filename)
            imgs = make_dataset(root, class_to_idx)
            np.savez_compressed(index_filename, **{'imgs': imgs})
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.load_in_mem = load_in_mem

        if self.load_in_mem:
            print('Loading all images into memory...')
            self.data, self.labels = [], []
            for index in tqdm(range(len(self.imgs))):
                path, target = imgs[index][0], imgs[index][1]
                self.data.append(self.transform(self.loader(path)))
                self.labels.append(target)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.load_in_mem:
            img = self.data[index]
            target = self.labels[index]
        else:
            path, target = self.imgs[index]
            img = self.loader(str(path))
            if self.transform is not None:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # print(img.size(), target)
        return img, int(target)

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


''' ILSVRC_HDF5: A dataset to support I/O from an HDF5 to avoid
    having to load individual images all the time. '''
import h5py as h5
import torch


class ILSVRC_HDF5(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None,
                 load_in_mem=False, train=True, download=False, validate_seed=0,
                 val_split=0, **kwargs):  # last four are dummies

        self.root = root
        self.num_imgs = len(h5.File(root, 'r')['labels'])

        # self.transform = transform
        self.target_transform = target_transform

        # Set the transform here
        self.transform = transform

        # load the entire dataset into memory?
        self.load_in_mem = load_in_mem

        # If loading into memory, do so now
        if self.load_in_mem:
            print('Loading %s into memory...' % root)
            with h5.File(root, 'r') as f:
                self.data = f['imgs'][:]
                self.labels = f['labels'][:]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # If loaded the entire dataset in RAM, get image from memory
        if self.load_in_mem:
            img = self.data[index]
            target = self.labels[index]

        # Else load it from disk
        else:
            with h5.File(self.root, 'r') as f:
                img = f['imgs'][index]
                target = f['labels'][index]

        img = ((torch.from_numpy(img).float() / 255) - 0.5) * 2

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, int(target)

    def __len__(self):
        return self.num_imgs

