# -*- coding: utf-8 -*-
import os
import time
import datetime
import copy
import numpy as np
import json
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.distributions import bernoulli
from utils import *
import tensorflow.compat.v1 as tf
import fid
import math
import functools
import inception_utils


class Trainer(object):
    def __init__(self, data_loader, config):
        # Fix seed
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)

        # Data loader
        self.data_loader = data_loader

        # arch and loss
        self.arch = config.arch
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel
        self.extra = config.extra

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.optim = config.optim
        self.lr_scheduler = config.lr_scheduler
        self.g_beta1 = config.g_beta1
        self.d_beta1 = config.d_beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model
        self.momentum = config.momentum

        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version
        self.backup_freq = config.backup_freq
        self.bup_path = config.bup_path
        self.metrics_path = config.metrics_path
        self.store_models_freq = config.store_models_freq

        # lookahead
        self.lookahead = config.lookahead
        self.lookahead_k = config.lookahead_k
        self.lookahead_super_slow_k = config.lookahead_super_slow_k
        self.lookahead_k_min = config.lookahead_k_min
        self.lookahead_k_max = config.lookahead_k_max
        self.lookahead_alpha = config.lookahead_alpha

        self.build_model()

        # imagenet
        if self.dataset == 'imagenet':
            z_ = inception_utils.prepare_z_(self.batch_size, self.z_dim, device='cuda', z_var=1.0)
            # Prepare Sample function for use with inception metrics
            self.sample_G_func = functools.partial(inception_utils.sample, G=self.G, z_=z_)
            self.sample_G_ema_func = functools.partial(inception_utils.sample, G=self.G_ema, z_=z_)
            self.sample_G_ema_slow_func = functools.partial(inception_utils.sample, G=self.G_ema_slow, z_=z_)
            # Prepare inception metrics: FID and IS
            self.get_inception_metrics = inception_utils.prepare_inception_metrics(dataset="./I32",
                                                                               parallel=False, no_fid=False)

        self.best_path = config.best_path  # dir for best-perf checkpoint

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

        self.info_logger = setup_logger(self.log_path)
        self.info_logger.info(config)
        self.cont = config.cont
        self.fid_freq = config.fid_freq

        if self.fid_freq > 0 and self.dataset != 'imagenet':
            self.fid_json_file = os.path.join(self.model_save_path, '../FID', 'fid.json')
            self.sample_size_fid = config.sample_size_fid
            if self.cont and os.path.isfile(self.fid_json_file):
                # load json files with fid scores
                self.fid_scores = load_json(self.fid_json_file)
            else:
                self.fid_scores = []
            sample_noise = torch.FloatTensor(self.sample_size_fid, self.z_dim).normal_()
            self.fid_noise_loader = torch.utils.data.DataLoader(sample_noise,
                                                                batch_size=200,
                                                                shuffle=False)
            # Inception Network
            _INCEPTION_PTH = fid.check_or_download_inception('./precalculated_statistics/inception-2015-12-05.pb')
            self.info_logger.info('Loading the Inception Network from: {}'.format(_INCEPTION_PTH))
            fid.create_inception_graph(_INCEPTION_PTH)  # load the graph into the current TF graph
            _gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
            # _gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
            self.fid_session = tf.Session(config=tf.ConfigProto(gpu_options=_gpu_options))
            self.info_logger.info('Loading real data FID stats from: {}'.format(config.fid_stats_path))
            _real_fid = np.load(config.fid_stats_path)
            self.mu_real, self.sigma_real = _real_fid['mu'][:], _real_fid['sigma'][:]
            _real_fid.close()
            make_folder(os.path.dirname(self.fid_json_file))
        elif self.fid_freq > 0:
            # make_folder(self.path)
            self.metrics_json_file = os.path.join(self.metrics_path, 'metrics.json')

    def train(self):
        self.data_gen = self._data_gen()

        # Fixed noise
        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))

        if self.cont:
            start = self.load_backup()
        else:
            start = 0

        if self.lr_scheduler > 0:
            _epoch = (start // len(self.data_loader)) if start > 0 else -1
            # Exponentially decaying learning rate
            self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer,
                                                                      gamma=self.lr_scheduler,
                                                                      last_epoch=_epoch)
            self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.d_optimizer,
                                                                      gamma=self.lr_scheduler,
                                                                      last_epoch=_epoch)
            self.scheduler_g_extra = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer_extra,
                                                                            gamma=self.lr_scheduler,
                                                                            last_epoch=_epoch)
            self.scheduler_d_extra = torch.optim.lr_scheduler.ExponentialLR(self.d_optimizer_extra,
                                                                            gamma=self.lr_scheduler,
                                                                            last_epoch=_epoch)

        # Start time
        start_time = time.time()
        for step in range(start, self.total_step):

            # ================= Train pair ================= #
            self._update_pair(step)

            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], Step [{}/{}]".format(elapsed, step + 1, self.total_step))

            # Sample images
            if (step + 1) % self.sample_step == 0:
                save_image(denorm(self.G(fixed_z).data),
                           os.path.join(self.sample_path, 'gen', 'iter%08d.png' % step))
                save_image(denorm(self.G_avg(fixed_z).data),
                           os.path.join(self.sample_path, 'gen_avg', 'iter%08d.png' % step))
                save_image(denorm(self.G_ema(fixed_z).data),
                           os.path.join(self.sample_path, 'gen_ema', 'iter%08d.png' % step))
                save_image(denorm(self.G_ema_slow(fixed_z).data),
                           os.path.join(self.sample_path, 'gen_ema_slow', 'iter%08d.png' % step))
            if self.model_save_step > 0 and (step+1) % self.model_save_step == 0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, 'gen', 'iter%08d.pth' % step))
                torch.save(self.G_avg.state_dict(),
                           os.path.join(self.model_save_path, 'gen_avg', 'iter%08d.pth' % step))
                torch.save(self.G_ema.state_dict(),
                           os.path.join(self.model_save_path, 'gen_ema', 'iter%08d.pth' % step))
                torch.save(self.G_ema_slow.state_dict(),
                           os.path.join(self.model_save_path, 'gen_ema_slow', 'iter%08d.pth' % step))
            if self.backup_freq > 0 and (step+1) % self.backup_freq == 0:
                self.backup(step)
            if self.store_models_freq > 0 and (step+1) % self.store_models_freq == 0:
                self.backup(step, store_step=True, update_timestamp=False)

            # If activated, compute Frechlet Inception Distance on the fly
            if self.fid_freq > 0 and (step + 1) % self.fid_freq == 0:
                if self.dataset == 'imagenet':
                    is_mean_ema_slow, is_std_ema_slow, fid_ema_slow = -1, -1, -1
                    is_mean, is_std, FID = self.get_inception_metrics(self.sample_G_func, 50000, num_splits=10)
                    print("Non-EMA metrics: Step [{}/{}], FID: {}, IS: {}/{}".format(
                        step + 1, self.total_step, FID, is_mean, is_std))
                    is_mean_ema, is_std_ema, FID_ema = self.get_inception_metrics(self.sample_G_ema_func, 50000,
                                                                                  num_splits=10)
                    print("EMA metrics: Step [{}/{}], FID: {}, IS: {}/{}".format(
                        step + 1, self.total_step, FID_ema, is_mean_ema, is_std_ema))
                    if self.lookahead and (step+1) % self.g_optimizer.k == 0:
                        if (self.lookahead_super_slow_k > 0 and (step+1) % self.lookahead_super_slow_k == 0) \
                                or self.lookahead_super_slow_k < 0:
                            is_mean_ema_slow, is_std_ema_slow, fid_ema_slow = self.get_inception_metrics(
                                self.sample_G_ema_slow_func, 50000, num_splits=10)
                            print("EMA slow metrics: Step [{}/{}], FID: {}, IS: {}/{}".format(
                                step + 1, self.total_step, fid_ema_slow, is_mean_ema_slow, is_std_ema_slow))
                    with open(self.metrics_json_file, 'a') as fs:
                        s = json.dumps(dict(itr=step+1,
                                       FID=float(FID),
                                       FID_ema=float(FID_ema),
                                       FID_ema_slow=float(fid_ema_slow),
                                       IS_mean=float(is_mean),
                                       IS_mean_ema_slow=float(is_mean_ema_slow),
                                       IS_std=float(is_std),
                                       IS_std_ema_slow=float(is_std_ema_slow),
                                       IS_mean_ema=float(is_mean_ema),
                                       IS_std_ema=float(is_std_ema)))
                        fs.write(f"{s}\n")
                else:
                    self.compute_fid_score(generator=self.G, timestamp=step)
                    self.info_logger.info("Step [{}/{}], FID: {}".format(
                        step + 1, self.total_step, self.fid_scores[-1]['fid']))

    def _data_gen(self):
        """ Data iterator

        :return: s
        """
        data_iter = iter(self.data_loader)
        while True:
            try:
                real_images, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                real_images, _ = next(data_iter)
            yield real_images

    def _update_pair(self, step):
        _lr_scheduler = self.lr_scheduler > 0 and step > 0 and step % len(self.data_loader) == 0
        self.D.train()
        self.G.train()

        if self.extra:
            self._extra_sync_nets()
            # ================== Train D @ t + 1/2 ================== #
            for _ in range(self.d_iters):
                real_images = tensor2var(next(self.data_gen))
                self._backprop_disc(D=self.D_extra, G=self.G, real_images=real_images,
                                    d_optim=self.d_optimizer_extra,
                                    scheduler_d=self.scheduler_d_extra if _lr_scheduler else None)

            # ================== Train G @ t + 1/2 ================== #
            self._backprop_gen(G=self.G_extra, D=self.D, bsize=self.batch_size,
                               g_optim=self.g_optimizer_extra,
                               scheduler_g=self.scheduler_g_extra if _lr_scheduler else None)

        # ================== Train D @ t + 1 ================== #
        for d_iter_idx in range(self.d_iters):
            real_images = tensor2var(next(self.data_gen))  # Re-sample
            d_loss_real = self._backprop_disc(G=self.G_extra, D=self.D, real_images=real_images,
                                              d_optim=self.d_optimizer,
                                              scheduler_d=self.scheduler_d if _lr_scheduler else None)

        # if self.lookahead and step % self.d_optimizer.k == 0: TODO: check Alt-LA-GAN
        #     self.d_optimizer.update_lookahead()

        # ================== Train G and gumbel @ t + 1 ================== #
        self._backprop_gen(G=self.G, D=self.D_extra, bsize=self.batch_size,
                           g_optim=self.g_optimizer,
                           scheduler_g=self.scheduler_g if _lr_scheduler else None)

        if self.lookahead and (step+1) % self.g_optimizer.k == 0:  # todo adjust code for different k
            self.d_optimizer.update_lookahead()
            self.g_optimizer.update_lookahead()
            if self.lookahead_super_slow_k > 0 and (step+1) % self.lookahead_super_slow_k == 0: #if super slow weights exist we just get ema on the super_slow
                self._update_avg_gen(step, net=self.G_avg_slow)
                self._update_ema_gen(net=self.G_ema_slow, beta_ema=0.9)
            else: # if we don't use super slow weights then we compute ema on the standard LA slow weights
                self._update_avg_gen(step, net=self.G_avg_slow)
                self._update_ema_gen(net=self.G_ema_slow)
        if self.lookahead and self.lookahead_super_slow_k > 0 and (step+1) % self.lookahead_super_slow_k == 0:
            print("super slow weights update !")
            self.d_optimizer.update_lookahead_super_slow()
            self.g_optimizer.update_lookahead_super_slow()

        # === Moving avg Generator-nets ===
        self._update_avg_gen(step)
        self._update_ema_gen()
        return d_loss_real

    def _backprop_disc(self, G, D, real_images, d_optim=None, scheduler_d=None):
        """Updates D (Vs. G).

        :param G:
        :param D:
        :param real_images:
        :param d_optim: if None, only backprop
        :return:
        """
        # Compute loss with real images
        # dr1, dr2, df1, df2, gf1, gf2 are attention scores
        d_out_real = D(real_images)
        if self.adv_loss == 'wgan-gp':
            d_loss_real = - torch.mean(d_out_real)
        elif self.adv_loss == 'hinge':
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        else:
            raise NotImplementedError

        # apply Gumbel Softmax
        z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
        fake_images = G(z)
        d_out_fake = D(fake_images)

        if self.adv_loss == 'wgan-gp':
            d_loss_fake = d_out_fake.mean()
        elif self.adv_loss == 'hinge':
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
        else:
            raise NotImplementedError

        # Backward + Optimize
        d_loss = d_loss_real + d_loss_fake
        if d_optim is not None:
            d_optim.zero_grad()
        d_loss.backward()
        
        if d_optim is not None:
            d_optim.step()
            if scheduler_d is not None:
                scheduler_d.step()

        if self.adv_loss == 'wgan-gp':  # todo: add SVRG
            # Compute gradient penalty
            alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
            interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
            out = D(interpolated)

            grad = torch.autograd.grad(outputs=out,
                                       inputs=interpolated,
                                       grad_outputs=torch.ones(out.size()).cuda(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

            # Backward + Optimize
            d_loss = self.lambda_gp * d_loss_gp

            if d_optim is not None:
                d_optim.reset_grad()
            d_loss.backward()
            if d_optim is not None:
                self.d_optimizer.step()
        return d_loss_real.data.item()

    def _backprop_gen(self, G, D, bsize, g_optim=True, scheduler_g=None):
        """Updates G (Vs. D).

        :param G:
        :param D:
        :param bsize:
        :param g_optim: if None only backprop
        :return:
        """
        z = tensor2var(torch.randn(bsize, self.z_dim))
        fake_images = G(z)

        g_out_fake = D(fake_images)  # batch x n
        if self.adv_loss == 'wgan-gp' or self.adv_loss == 'hinge':
            g_loss_fake = - g_out_fake.mean()

        if g_optim is not None:
            g_optim.zero_grad()
        g_loss_fake.backward()
        if g_optim is not None:
            g_optim.step()
            if scheduler_g is not None:
                scheduler_g.step()
        return g_loss_fake.data.item()

    def build_model(self):
        # Models                    ###################################################################
        if self.arch == 'resnet':
            from models import resnet_models
            self.G = resnet_models.Generator(self.z_dim).cuda()
            self.D = resnet_models.Discriminator().cuda()
        else:
            raise NotImplementedError

        if self.extra:
            self.G_extra = copy.deepcopy(self.G).cuda()
            self.D_extra = copy.deepcopy(self.D).cuda()
        else:
            self.G_extra = self.G
            self.D_extra = self.D

        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
            if self.extra:
                self.G_extra = nn.DataParallel(self.G_extra)
                self.D_extra = nn.DataParallel(self.D_extra)

        self.G_avg = copy.deepcopy(self.G)
        self.G_ema = copy.deepcopy(self.G)
        self.G_avg_slow = copy.deepcopy(self.G)
        self.G_ema_slow = copy.deepcopy(self.G)
        self._requires_grad(self.G_avg, False)
        self._requires_grad(self.G_ema, False)
        self._requires_grad(self.G_avg_slow, False)
        self._requires_grad(self.G_ema_slow, False)

        # Logs, Loss & optimizers   ###################################################################
        # grad_var_logger_g = setup_logger(self.log_path, 'log_grad_var_g.log')
        # grad_var_logger_d = setup_logger(self.log_path, 'log_grad_var_d.log')
        # grad_mean_logger_g = setup_logger(self.log_path, 'log_grad_mean_g.log')
        # grad_mean_logger_d = setup_logger(self.log_path, 'log_grad_mean_d.log')
        # grad_var_logger_g = grad_var_logger_d = grad_mean_logger_g = grad_mean_logger_d = None

        if self.optim == 'sgd':
            from optim import sgd as optm
            self.g_optimizer = optm.SGD(filter(lambda p: p.requires_grad, self.G.parameters()),
                                         self.g_lr, momentum=self.momentum)
            self.d_optimizer = optm.SGD(filter(lambda p: p.requires_grad, self.D.parameters()),
                                         self.d_lr, momentum=self.momentum)
            if self.extra:
                self.g_optimizer_extra = optm.SGD(filter(lambda p: p.requires_grad,
                                                          self.G_extra.parameters()),
                                                   self.g_lr)
                self.d_optimizer_extra = optm.SGD(filter(lambda p: p.requires_grad,
                                                          self.D_extra.parameters()),
                                                   self.d_lr)
            else:
                self.g_optimizer_extra = self.g_optimizer
                self.d_optimizer_extra = self.d_optimizer
        elif self.optim == 'adam':
            from optim import adam as optm
            self.g_optimizer = optm.Adam(filter(lambda p: p.requires_grad, self.G.parameters()),
                                          self.g_lr, [self.g_beta1, self.beta2])
            self.d_optimizer = optm.Adam(filter(lambda p: p.requires_grad, self.D.parameters()),
                                          self.d_lr, [self.d_beta1, self.beta2])
            if self.extra:
                self.g_optimizer_extra = optm.Adam(filter(lambda p: p.requires_grad,
                                                           self.G_extra.parameters()),
                                                    self.g_lr, [self.g_beta1, self.beta2])
                self.d_optimizer_extra = optm.Adam(filter(lambda p: p.requires_grad,
                                                           self.D_extra.parameters()),
                                                    self.d_lr, [self.d_beta1, self.beta2])
            else:
                self.g_optimizer_extra = self.g_optimizer
                self.d_optimizer_extra = self.d_optimizer
        elif self.optim == 'radam':
            from optim import radam as optm
            self.g_optimizer = optm.RAdam(filter(lambda p: p.requires_grad, self.G.parameters()),
                                          self.g_lr, [self.g_beta1, self.beta2])
            self.d_optimizer = optm.RAdam(filter(lambda p: p.requires_grad, self.D.parameters()),
                                          self.d_lr, [self.d_beta1, self.beta2])
            if self.extra:
                self.g_optimizer_extra = optm.RAdam(filter(lambda p: p.requires_grad,
                                                           self.G_extra.parameters()),
                                                    self.g_lr, [self.g_beta1, self.beta2])
                self.d_optimizer_extra = optm.RAdam(filter(lambda p: p.requires_grad,
                                                           self.D_extra.parameters()),
                                                    self.d_lr, [self.d_beta1, self.beta2])
            else:
                self.g_optimizer_extra = self.g_optimizer
                self.d_optimizer_extra = self.d_optimizer
        else:
            raise NotImplementedError('Supported optimizers: SGD, Adam, VRAd')
        if self.lookahead:
            from optim import lookahead as la
            self.g_optimizer = la.Lookahead(self.g_optimizer, k=self.lookahead_k,
                                            alpha=self.lookahead_alpha,
                                            k_min=self.lookahead_k_min,
                                            k_max=self.lookahead_k_max,
                                            super_slow_k=self.lookahead_super_slow_k)
            self.d_optimizer = la.Lookahead(self.d_optimizer, k=self.lookahead_k,
                                            alpha=self.lookahead_alpha,
                                            k_min=self.lookahead_k_min,
                                            k_max=self.lookahead_k_max,
                                            super_slow_k=self.lookahead_super_slow_k)

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def _extra_sync_nets(self):
        """ Helper function. Copies the current parameters to the t+1/2 parameters,
         stored as 'net' and 'extra_net', respectively.

        :return: [None]
        """
        self.G_extra.load_state_dict(self.G.state_dict())
        self.D_extra.load_state_dict(self.D.state_dict())

    @staticmethod
    def _update_avg(avg_net, net, avg_step):
        """Updates average network."""
        # Todo: input val
        net_param = list(net.parameters())
        for i, p in enumerate(avg_net.parameters()):
            with torch.no_grad():
                p.mul_((avg_step - 1) / avg_step)
                p.add_(net_param[i].div(avg_step))

    @staticmethod
    def _requires_grad(_net, _bool=True):
        """Helper function which sets the requires_grad of _net to _bool.

        Raises:
            TypeError: _net is given but is not derived from nn.Module, or
                       _bool is not boolean

        :param _net: [nn.Module]
        :param _bool: [bool, optional] Default: True
        :return: [None]
        """
        if _net and not isinstance(_net, torch.nn.Module):
            raise TypeError("Expected torch.nn.Module. Got: {}".format(type(_net)))
        if not isinstance(_bool, bool):
            raise TypeError("Expected bool. Got: {}".format(type(_bool)))

        if _net is not None:
            for _w in _net.parameters():
                _w.requires_grad = _bool

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))

    def backup(self, iteration, store_step=False, update_timestamp=True, dir=None):
        """Back-ups the networks & optimizers' states.

        Note: self.g_extra & self.d_extra are not stored, as these are copied from
        self.G & self.D at the beginning of each iteration. However, the optimizers
        are backed up.

        :param iteration: [int]
        :param store_step: [bool] store only disc and gen, with the iteration as a postfix
        :param update_timestamp: [bool]
        :return: [None]
        """
        path = self.bup_path if dir is None else dir
        if store_step:
            torch.save(self.G.state_dict(), os.path.join(path, 'gen_%d.pth' % iteration))
            torch.save(self.D.state_dict(), os.path.join(path, 'disc_%d.pth' % iteration))
        else:
            torch.save(self.G.state_dict(), os.path.join(path, 'gen.pth'))
            torch.save(self.D.state_dict(), os.path.join(path, 'disc.pth'))
            torch.save(self.G_avg.state_dict(), os.path.join(path, 'gen_avg.pth'))
            torch.save(self.G_ema.state_dict(), os.path.join(path, 'gen_ema.pth'))
            torch.save(self.G_avg_slow.state_dict(), os.path.join(path, 'gen_avg_slow.pth'))
            torch.save(self.G_ema_slow.state_dict(), os.path.join(path, 'gen_ema_slow.pth'))

            torch.save(self.g_optimizer.state_dict(), os.path.join(path, 'gen_optim.pth'))
            torch.save(self.d_optimizer.state_dict(), os.path.join(path, 'disc_optim.pth'))
            torch.save(self.g_optimizer_extra.state_dict(), os.path.join(path, 'gen_extra_optim.pth'))
            torch.save(self.d_optimizer_extra.state_dict(), os.path.join(path, 'disc_extra_optim.pth'))

        if update_timestamp:
            with open(os.path.join(path, "timestamp.txt"), "w") as fff:
                fff.write("%d" % iteration)

    def load_backup(self):
        """Loads the Backed-up networks & optimizers' states.

        Note: self.g_extra & self.d_extra are not stored, as these are copied from
        self.G & self.D at the beginning of each iteration. However, the optimizers
        are backed up.

        :return: [int] timestamp to continue from
        """
        if not os.path.exists(self.bup_path):
            raise ValueError('Cannot load back-up. Directory {} '
                             'does not exist.'.format(self.bup_path))

        self.G.load_state_dict(torch.load(os.path.join(self.bup_path, 'gen.pth')))
        self.D.load_state_dict(torch.load(os.path.join(self.bup_path, 'disc.pth')))
        self.G_avg.load_state_dict(torch.load(os.path.join(self.bup_path, 'gen_avg.pth')))
        self.G_ema.load_state_dict(torch.load(os.path.join(self.bup_path, 'gen_ema.pth')))
        self.G_avg_slow.load_state_dict(torch.load(os.path.join(self.bup_path, 'gen_avg_slow.pth')))
        self.G_ema_slow.load_state_dict(torch.load(os.path.join(self.bup_path, 'gen_ema_slow.pth')))

        self.g_optimizer.load_state_dict(torch.load(os.path.join(self.bup_path, 'gen_optim.pth')))
        self.d_optimizer.load_state_dict(torch.load(os.path.join(self.bup_path, 'disc_optim.pth')))
        self.g_optimizer_extra.load_state_dict(torch.load(os.path.join(self.bup_path, 'gen_extra_optim.pth')))
        self.d_optimizer_extra.load_state_dict(torch.load(os.path.join(self.bup_path, 'disc_extra_optim.pth')))

        with open(os.path.join(self.bup_path, "timestamp.txt"), "r") as fff:
            timestamp = [int(x) for x in next(fff).split()]  # read first line
            if not len(timestamp) == 1:
                raise ValueError('Could not determine timestamp of the backed-up models.')
            timestamp = int(timestamp[0]) + 1

        self.info_logger.info("Loaded models from %s, at timestamp %d." %
                              (self.bup_path, timestamp))
        return timestamp

    def _update_avg_gen(self, n_gen_update, net=None):
        """ Updates the uniform average generator. """
        net = net or self.G_avg  # can also be called with net=self.G_avg_slow
        l_param = list(self.G.parameters())
        l_avg_param = list(net.parameters())
        if len(l_param) != len(l_avg_param):
            raise ValueError("Got different lengths: {}, {}".format(len(l_param), len(l_avg_param)))

        for i in range(len(l_param)):
            with torch.no_grad():
                l_avg_param[i].data.copy_(l_avg_param[i].data.mul(n_gen_update).div(n_gen_update + 1.).add(
                                          l_param[i].data.div(n_gen_update + 1.)))

    def _update_ema_gen(self, beta_ema=0.9999, net=None):
        """ Updates the exponential moving average generator. """
        net = net or self.G_ema  # can also be called with net=self.G_ema_slow
        l_param = list(self.G.parameters())
        l_ema_param = list(net.parameters())
        if len(l_param) != len(l_ema_param):
            raise ValueError("Got different lengths: {}, {}".format(len(l_param), len(l_ema_param)))

        for i in range(len(l_param)):
            with torch.no_grad():
                l_ema_param[i].data.copy_(l_ema_param[i].data.mul(beta_ema).add(
                    l_param[i].data.mul(1-beta_ema)))
        # gen_param_ema[j] = gen_param_ema[j] * BETA_EMA + param.data.clone() * (1 -  beta_ema)

    def compute_fid_score(self, generator, timestamp):
        """
        Computes FID of generator using fixed noise dataset;
        appends the current score to the list of computed scores;
        and overwrites the json file that logs the fid scores.

        :param generator: [nn.Module]
        :param timestamp: [int]
        :return: None
        """
        generator.eval()
        fake_samples = np.empty((self.sample_size_fid, self.imsize, self.imsize, 3))
        for j, noise in enumerate(self.fid_noise_loader):
            noise = noise.cuda()
            i1 = j * 200  # batch_size = 200
            i2 = i1 + noise.size(0)
            samples = generator(noise).cpu().data.add(1).mul(255 / 2.0)
            fake_samples[i1:i2] = samples.permute(0, 2, 3, 1).numpy()
        generator.train()
        mu_g, sigma_g = fid.calculate_activation_statistics(fake_samples, self.fid_session, batch_size=100)
        fid_score = fid.calculate_frechet_distance(mu_g, sigma_g, self.mu_real, self.sigma_real)
        _result = {
                   'entry': len(self.fid_scores),
                   'iter': timestamp,
                   'fid': fid_score}

        # if best update the checkpoint in self.best_path
        new_best = True
        for prev_fid in self.fid_scores:
            if prev_fid['fid'] < fid_score:
                new_best = False
                break
        if new_best:
            self.backup(timestamp, dir=self.best_path)

        self.fid_scores.append(_result)
        with open(self.fid_json_file, 'w') as _f_fid:
            json.dump(self.fid_scores, _f_fid, sort_keys=True, indent=4, separators=(',', ': '))

