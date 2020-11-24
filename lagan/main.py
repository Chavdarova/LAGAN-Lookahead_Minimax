# -*- coding: utf-8 -*-
"""
Implementation of the Lookahead-Minmax algorithm.
See: Taming GANs with Lookahead-Minmax, 2020.

USAGE:
    run `python main.py --help` for available options.
"""
from parameter import *
from trainer import Trainer
from data_loader import Data_Loader
from torch.backends import cudnn
from utils import make_folder


def main(config):
    cudnn.benchmark = True

    print("Loading data...")
    data_loader = Data_Loader(config.train, config.dataset, config.imsize,
                             config.batch_size, config.image_path, shuf=config.train)
    print('Done.')

    # Create directories if these do not exist
    for _subdir in ['gen', 'gen_avg', 'gen_ema', 'gen_ema_slow']:
        make_folder(config.model_save_path, _subdir)
        make_folder(config.sample_path, _subdir)
    make_folder(config.log_path)
    make_folder(config.best_path)
    if config.backup_freq > 0:
        make_folder(config.bup_path)
    if config.dataset == 'imagenet' and config.fid_freq > 0:
        make_folder(config.metrics_path)

    # Train
    trainer = Trainer(data_loader.loader(), config)
    trainer.train()


if __name__ == '__main__':
    args = get_parameters()
    print(args)
    main(args)
