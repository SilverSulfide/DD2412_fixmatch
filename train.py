import os
import logging
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import sampler, DataLoader
from torch.utils.data.sampler import BatchSampler

from utils import TBLog, net_builder, get_logger, count_parameters
from fixmatch import FixMatch
from ssl_dataset import SSL_Dataset
from haparams import create_hparams

from progressbar import ProgressBar


def main(args, hps):
    # random seed has to be set for the synchronization of labeled data sampling in each process.
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

    # SET Devices for (Distributed) DataParallel
    if not torch.cuda.is_available():
        raise Exception('ONLY GPU TRAINING IS SUPPORTED')

    else:
        torch.cuda.set_device(0)
        model.train_model = model.train_model.cuda()
        model.eval_model = model.eval_model.cuda()

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

    # If args.resume, load checkpoints from args.load_path
    if args.resume:
        model.load_model(args.load_path)

    # START TRAINING of FixMatch

    # start resnet training
    model.train_model.train()

    # start_batch.record()
    best_eval_acc, best_it = 0.0, 0

    # use progressbar in between model saves
    pbar = ProgressBar(maxval=hps.train.log_interval)
    pbar.start()
    progress = 0

    for (x_lb, y_lb), (x_ulb_w, x_ulb_s, _) in zip(loader_dict['train_lb'], loader_dict['train_ulb']):

        # update the progressbar
        pbar.update(progress)

        # prevent the training iterations from exceeding num_train_iter
        if model.it > hps.train.num_train_iters:
            break

        # set amount of labelled samples
        model.init_lb(x_lb)

        x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(), x_ulb_w.cuda(), x_ulb_s.cuda()
        y_lb = y_lb.long().cuda()  # FIXME: windows ghetto fix
        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        from torch.cuda.amp import autocast

        # inference, loss, update
        with autocast():
            # inference
            lb_logits, ulb_w_logits, ulb_s_logits = model.forward(inputs)

            # loss
            sup_loss, unsup_loss, total_loss = model.loss(lb_logits, y_lb, ulb_w_logits, ulb_s_logits)

            # backprop
            model.backward(total_loss)

        # update exponential moving avarage
        model._eval_model_update()

        # tensorboard_dict update
        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.detach()
        tb_dict['train/unsup_loss'] = unsup_loss.detach()
        tb_dict['train/total_loss'] = total_loss.detach()
        tb_dict['lr'] = model.optimizer.param_groups[0]['lr']

        if model.it % hps.train.log_interval == 0:
            # stop the progress bar
            pbar.finish()
            eval_dict = model.evaluate(loader_dict['eval'])

            tb_dict.update(eval_dict)

            save_path = os.path.join(args.save_dir, args.save_name)

            if tb_dict['eval/top-1-acc'] > best_eval_acc:
                best_eval_acc = tb_dict['eval/top-1-acc']
                best_it = model.it

            model.print_fn(
                f"{model.it} iteration, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")

            # reset progressbar
            progress = 0
            pbar = ProgressBar(maxval=hps.train.log_interval)
            pbar.start()

        if model.it == best_it:
            model.save_model('model_best.pth', save_path)

        model.tb_log.update(tb_dict, model.it)

        # update iteration
        model.update_iter()
        progress += 1

        # free memory
        del tb_dict

    eval_dict = model.evaluate()
    eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
    # eval_dict

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
