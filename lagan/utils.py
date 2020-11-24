# -*- coding: utf-8 -*-
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import copy


def make_folder(path, version=None):
    path = os.path.join(path, version) if version else path
    if not os.path.exists(path):
        os.makedirs(path)


def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)


def var2tensor(x):
    return x.data.cpu()


def var2numpy(x):
    return x.data.cpu().numpy()


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def setup_logger(out_dir, file_name='log.log'):
    """
    Sets up an info logger.

    Raises:
        NotADirectoryError: _data_root is not a directory
    :param out_dir: [str] directory to store the log file
    :param file_name: [str]
    :return: [module]
    """
    if not os.path.exists(out_dir):
        raise NotADirectoryError("Could not open {}".format(out_dir))
    import logging
    logger = logging.getLogger(os.path.splitext(os.path.basename(file_name))[0])
    handler = logging.FileHandler(os.path.join(out_dir, file_name))
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def load_json(jsonf):
    if not os.path.isfile(jsonf):
        raise FileNotFoundError
    import json
    try:
        with open(jsonf) as data_file:
            data = json.load(data_file)
    except ValueError:
        raise ValueError("File exists, but content is not json")
    return data

