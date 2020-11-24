# -*- coding: utf-8 -*-
import os
import torch
import torchvision.datasets as dsets
from torchvision import transforms
from torch.utils.data import DataLoader


class Data_Loader():
    def __init__(self, train, dataset, image_size, batch_size, image_path=None, shuf=True):
        self.dataset = dataset
        self.imsize = image_size
        self.batch = batch_size
        self.image_path = image_path
        self.shuf = shuf
        self.train = train

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize, self.imsize)))  # new/old version: Resize/Scale
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes='bedroom_train'):  # church_outdoor_train
        transforms = self.transform(True, True, True, False)
        if self.image_path is None:
            raise NotImplementedError("LSUN is not downloaded.")
        dataset = dsets.LSUN(self.image_path, classes=[classes], transform=transforms)
        return dataset

    def load_celeb(self):
        transforms = self.transform(True, True, True, True)
        dataroot = self.image_path or os.path.expanduser('~/datasets/CelebA/')  # subdir img_align_celeba, 178x218
        dataset = dsets.ImageFolder(dataroot, transform=transforms)
        return dataset

    def load_imagenet(self):
        """ Assumes a file `ILSVRC32.hdf5` exists in the current directory.
        """

        dataset = 'I32_hdf5'
        num_workers = 8
        pin_memory = True
        load_in_mem = True
        use_multiepoch_sampler = False

        import data_imagenet as dset
        dset_dict = {'I32_hdf5': dset.ILSVRC_HDF5}
        data_root = './%s' % 'ILSVRC32.hdf5'
        print('Using dataset location %s' % data_root)
        which_dataset = dset_dict[dataset]

        dataset_kwargs = {'index_filename': '%s_imgs.npz' % dataset}

        # HDF5 datasets have their own inbuilt transform, no need to train_transform
        if 'hdf5' in dataset:
            train_transform = None
        else:
            raise NotImplementedError('only imagenet with hdf5 file is implemented')

        train_set = which_dataset(root=data_root, transform=train_transform,
                                  load_in_mem=load_in_mem, **dataset_kwargs)
        loaders = []
        if use_multiepoch_sampler:
            print('Using multiepoch sampler from start_itr %d...' % start_itr)
            loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
            sampler = MultiEpochSampler(train_set, num_epochs, start_itr, batch_size)
            train_loader = DataLoader(train_set, batch_size=batch_size,
                                      sampler=sampler, **loader_kwargs)
        else:
            loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                             'drop_last': True}  # Default, drop last incomplete batch
            train_loader = DataLoader(train_set, batch_size=self.batch,
                                      shuffle=self.shuf, **loader_kwargs)
        loaders.append(train_loader)
        print('len of loaders %d, total minibatches %d, total samples %d' % (len(loaders),
                                    len(loaders[0]), len(loaders[0])*self.batch))
        return loaders[0]

    def load_cifar(self):
        transforms = self.transform(True, True, True, False)
        dataroot = self.image_path or os.path.expanduser('~/datasets/cifar10/')
        dataset = dsets.CIFAR10(dataroot, transform=transforms, download=True)
        return dataset

    def load_stl10(self):
        transforms = self.transform(True, True, True, False)
        dataroot = self.image_path or os.path.expanduser('~/datasets/STL10/')
        dataset = dsets.STL10(dataroot, transform=transforms, download=True)
        return dataset

    def load_svhn(self):
        transforms = self.transform(True, True, True, False)
        dataroot = self.image_path or os.path.expanduser('~/datasets/svhn/')
        dataset = dsets.SVHN(dataroot, transform=transforms, download=True)
        return dataset

    def loader(self):
        if self.dataset == 'lsun':
            dataset = self.load_lsun()
        elif self.dataset == 'celeba':
            dataset = self.load_celeb()
        elif self.dataset == 'imagenet':
            dataset = self.load_imagenet()
        elif self.dataset == 'cifar10':
            dataset = self.load_cifar()
        elif self.dataset == 'stl10':
            dataset = self.load_stl10()
        elif self.dataset == 'svhn':
            dataset = self.load_svhn()
        else:
            raise NotImplementedError('Available datasets: lsun, celeba, imagenet, '
                                      'cifar10, stl10 and svhn.')

        if self.dataset == 'imagenet':
            return dataset

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch,
                                             shuffle=self.shuf,
                                             num_workers=2,
                                             drop_last=True)
        return loader

