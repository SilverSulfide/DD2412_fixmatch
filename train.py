import os
import logging
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import sampler, DataLoader
from torch.utils.data.sampler import BatchSampler

from train_utils import TBLog, get_SGD, get_cosine_schedule_with_warmup,net_builder, get_logger, count_parameters
from fixmatch import FixMatch
from ssl_dataset import SSL_Dataset
from haparams import create_hparams


def main(args, hps):
    global best_acc1

    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    # logger_level = "WARNING"
    # tb_log = None

    tb_log = TBLog(save_path, 'tensorboard')
    logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {0} for training")

    # SET FixMatch: class FixMatch in models.fixmatch
    bn_momentum = 1.0 - hps.train.ema_m
    resnet_builder = net_builder({'depth': hps.model.depth,
                                  'widen_factor': hps.model.widen_factor,
                                  'leaky_slope': hps.model.leaky_slope,
                                  'bn_momentum': bn_momentum,
                                  'dropRate': hps.model.dropout})

    model = FixMatch(resnet_builder,
                     hps,
                     tb_log=tb_log,
                     logger=logger)

    logger.info(f'Number of Trainable Params: {count_parameters(model.train_model)}')

    # SET Optimizer & LR Scheduler
    # construct SGD and cosine lr scheduler
    optimizer = get_SGD(model.train_model, 'SGD', hps.train.learning_rate, hps.train.momentum, hps.train.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                hps.train.num_train_iters,
                                                num_warmup_steps=hps.train.num_train_iters * 0)
    # set SGD and cosine lr on FixMatch
    model.set_optimizer(optimizer, scheduler)

    # SET Devices for (Distributed) DataParallel
    if not torch.cuda.is_available():
        raise Exception('ONLY GPU TRAINING IS SUPPORTED')

    else:
        torch.cuda.set_device(0)
        model.train_model = model.train_model.cuda()
        model.eval_model = model.eval_model.cuda()

    logger.info(f"model_arch: {model}")
    logger.info(f"Arguments: {args}")

    cudnn.benchmark = True

    # Construct Datasets
    # training sets for labelled and unlabelled
    lb_dset, ulb_dset = SSL_Dataset(name=hps.data.dataset, train=True,
                                    num_classes=hps.data.num_classes, data_dir=hps.data.data_dir,
                                    args=args).get_ssl_dset(hps.train.num_labels)

    # evaluation set
    eval_dset = SSL_Dataset(name=hps.data.dataset, train=False,
                            num_classes=hps.data.num_classes, data_dir=hps.data.data_dir, args=args).get_dset()

    loader_dict = {}
    dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset}

    # Construct labelled data loader
    data_sampler = getattr(torch.utils.data.sampler, hps.data.data_sampler)
    num_samples = hps.train.batch * hps.train.num_train_iters

    data_sampler = data_sampler(dset_dict['train_lb'], replacement=True, num_samples=num_samples, generator=None)

    batch_sampler = BatchSampler(data_sampler, batch_size=hps.train.batch, drop_last=True)

    loader_dict['train_lb'] = DataLoader(dset_dict['train_lb'], batch_sampler=batch_sampler,
                                         num_workers=1, pin_memory=True)

    # Construct unlabelled data loader using corresponding data ratio
    data_sampler = getattr(torch.utils.data.sampler, hps.data.data_sampler)
    num_samples = hps.train.batch * hps.train.label_ratio * hps.train.num_train_iters

    data_sampler = data_sampler(dset_dict['train_ulb'], replacement=True, num_samples=num_samples, generator=None)

    batch_sampler = BatchSampler(data_sampler, batch_size=hps.train.batch * hps.train.label_ratio, drop_last=True)

    loader_dict['train_ulb'] = DataLoader(dset_dict['train_ulb'], batch_sampler=batch_sampler,
                                          num_workers=4, pin_memory=True)

    # Construct evaluation data loader
    loader_dict['eval'] = DataLoader(dset_dict['eval'], batch_size=hps.train.eval_batch, shuffle=False,
                                     num_workers=1, pin_memory=True)

    # set DataLoader on FixMatch
    model.set_data_loader(loader_dict)

    # If args.resume, load checkpoints from args.load_path
    if args.resume:
        model.load_model(args.load_path)

    # START TRAINING of FixMatch
    trainer = model.train
    trainer(args, hps, logger=logger)

    model.save_model('latest_model.pth', save_path)

    logging.warning(f"GPU {0} training is FINISHED")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--save_name', type=str, default='fixmatch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
    parser.add_argument('-hp', '--hparams', type=str,
                        required=True, help='path to model parameters')
    # Experimental arguments
    parser.add_argument('-tr', '--translate', action='store_true', help='Add translation transform to weak augment')
    parser.add_argument('-n', '--noise', action='store_true', help='Add noise transform to weak augment')
    parser.add_argument('-c', '--contrast', action='store_true', help='Add contrast transform to weak augment')

    args = parser.parse_args()
    hps = create_hparams(args.hparams)
    print("Hyperparameter settings: ")
    for group in vars(hps):
        print(group, getattr(hps, group))

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                                If you want over-write, give --overwrite in the argument.')

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main(args, hps)
