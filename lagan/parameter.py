# -*- coding: utf-8 -*-
import argparse
import os


def str2bool(v):
    return v.lower() in ('true')


def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--arch', type=str, default='resnet',
                        choices=['resnet'])
    parser.add_argument('--adv_loss', type=str, default='hinge', choices=['wgan-gp', 'hinge'])
    parser.add_argument('--imsize', type=int, default=32)
    parser.add_argument('--g_num', type=int, default=5)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--version', type=str,
                        help='name of sub-directory: ./results/<dataset>/<version>_optim/'
                             '{data, logs, models, samples, attn, backup}')

    # Training setting
    parser.add_argument('--total_step', type=int, default=500000,
                        help='how many times to update the generator, default: %(default)s')
    parser.add_argument('--d_iters', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128,  help='batch-size, default: %(default)s')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0004)
    parser.add_argument('--g_beta1', type=float, default=0.0, help='valid only for (R)Adam')
    parser.add_argument('--d_beta1', type=float, default=0.0, help='valid only for (R)Adam')
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--momentum', type=float, default=0, help='used only if --optim is sgd')

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='celeba',
                        choices=['lsun', 'celeba', 'imagenet', 'cifar10', 'lsun', 'svhn'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    parser.add_argument('--cont', action='store_true', help='continue training, default: %(default)s')

    # lookahead
    parser.add_argument('--lookahead', type=str2bool, default=False)
    parser.add_argument('--lookahead_k', type=int, default=5, help='if -1 random value in [3,1000]')
    parser.add_argument('--lookahead_super_slow_k', type=int, default=-1, help='not used if -1')
    parser.add_argument('--lookahead_k_min', type=int, default=3, help='valid if k<=0')
    parser.add_argument('--lookahead_k_max', type=int, default=1000, help='valid if k<=0')
    parser.add_argument('--lookahead_alpha', type=float, default=.5)

    # Path
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--log_path', type=str, default='./results/<dataset>/<version>/logs')
    parser.add_argument('--model_save_path', type=str, default='./results/<dataset>/<version>/models')
    parser.add_argument('--sample_path', type=str, default='./results/<dataset>/<version>/samples')
    parser.add_argument('--best_path', type=str, default='./results/<dataset>/<version>/best')
    parser.add_argument('--bup_path', type=str, default='./results/<dataset>/<version>/backup')
    parser.add_argument('--metrics_path', type=str, default='./results/<dataset>/<version>/metrics')

    # Step size
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000,
                        help='frequency to store generators for metrics, use <=0 to cancel storing, '
                             'default: %(default)s')
    parser.add_argument('--store_models_freq', type=int, default=0)
    parser.add_argument('--backup_freq', type=int, default=10000,
                        help='Frequency to backup models and optim states, default: %(default)s')

    # Random seed
    parser.add_argument('--seed', type=int, default=1, help='fixed random seed, default: %(default)s')

    parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'adam', 'radam'])
    parser.add_argument('--extra', type=str2bool, default=False)
    parser.add_argument('--lr_scheduler', type=float, default=-1,
                        help='Gamma parameter for a learning rate scheduler. Use negative value to cancel it. '
                             'Default: %(default)s')

    # compute fid on the fly
    parser.add_argument('--fid_freq', help='if fid_freq > 0, FID is computed every fid_freq-th iteration, '
                                           'default: %(default)s',
                        default=10000, type=int)
    parser.add_argument('--sample_size_fid', type=int, default=50000,
                        help="Sample size of FID, default: %(default)s")
    parser.add_argument('--fid_stats_path', help='pre-calculated fid stats for the real data')

    _args = parser.parse_args()
    if not _args.version:
        # `extra` or `gan`
        _args.version = 'extra' if _args.extra else 'gan'
        _args.version += '_' + _args.optim
        _args.version += '_' + _args.arch

    _args.version += '/G%f_D%f' % (_args.g_lr, _args.d_lr)
    if _args.optim != 'sgd':
        _args.version += '_beta_%.1f' % _args.g_beta1
    if _args.lr_scheduler > 0:
        _args.version += '_gamma_%.2f' % _args.lr_scheduler
    if _args.batch_size != 64:
        _args.version += '_bs_%d' % _args.batch_size
    if _args.d_iters != 1:
        _args.version += '_dIters_%d' % _args.d_iters
    if _args.store_models_freq > 0:
        _args.version += '_store_%d' % _args.store_models_freq

    if _args.momentum > 0 and _args.optim == 'sgd':
        _args.version += '_m_%.2f' % _args.momentum
    if _args.lookahead is not None and _args.lookahead is True:
        print('Using lookahead.')
        _args.version += '_{}lookahead{}_ssk{}'.format(_args.lookahead_k,
                                                       _args.lookahead_alpha, 
                                                       _args.lookahead_super_slow_k)
    else:
        print('Without lookahead.')

    if _args.seed != 1:
        _args.version += '_seed{}'.format(_args.seed)

    _args.log_path = _args.log_path.replace("<dataset>", _args.dataset)
    _args.model_save_path = _args.model_save_path.replace("<dataset>", _args.dataset)
    _args.sample_path = _args.sample_path.replace("<dataset>", _args.dataset)
    _args.best_path = _args.best_path.replace("<dataset>", _args.dataset)
    _args.bup_path = _args.bup_path.replace("<dataset>", _args.dataset)
    _args.metrics_path = _args.metrics_path.replace("<dataset>", _args.dataset)

    _args.log_path = _args.log_path.replace("<version>", _args.version)
    _args.model_save_path = _args.model_save_path.replace("<version>", _args.version)
    _args.sample_path = _args.sample_path.replace("<version>", _args.version)
    _args.best_path = _args.best_path.replace("<version>", _args.version)
    _args.bup_path = _args.bup_path.replace("<version>", _args.version)
    _args.metrics_path = _args.metrics_path.replace("<version>", _args.version)

    if _args.dataset != 'imagenet':
        if _args.fid_freq > 0 and not _args.fid_stats_path:
            if 'cifar10' in _args.dataset:
                _args.fid_stats_path = 'precalculated_statistics/fid_stats_cifar10_train.npz'
            elif 'celeba' in _args.dataset:
                _args.fid_stats_path = 'precalculated_statistics/fid_stats_celeba.npz'
            elif 'lsun' in _args.dataset:
                _args.fid_stats_path = 'precalculated_statistics/fid_stats_lsun_train.npz'
            elif 'stl10' in _args.dataset:
                _args.fid_stats_path = 'precalculated_statistics/fid_stats_stl10_train.npz'
            elif 'svhn' in _args.dataset:
                _args.fid_stats_path = 'precalculated_statistics/fid_stats_svhn_train.npz'
            else:
                raise ValueError('Unknown path to the pre-calculated fid statistics.')
        if _args.fid_freq > 0 and not os.path.isfile(_args.fid_stats_path):
            raise FileNotFoundError('Could not find {}'.format(_args.fid_stats_path))
    return _args
