import torch
from torch.utils.tensorboard import SummaryWriter
import os
import logging
import numpy as np


def init_optimiser(resnet, hps):
    """
    :param resnet: pytorch resnet class
    :param hps: hyperameters
    :return: optimiser object
    """
    # https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/2
    # remove weight decay from from layers having batch norm

    decay = []
    no_decay = []
    for name, param in resnet.named_parameters():

        # proceed only if the parameter requires gradient
        if not param.requires_grad:
            continue
        if 'bn' in name:
            no_decay.append(param)
        else:
            decay.append(param)

    per_param_args = [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay}]

    return torch.optim.SGD(per_param_args, lr=hps.train.learning_rate, momentum=hps.train.momentum,
                           weight_decay=hps.train.weight_decay, nesterov=True)


def init_lr_scheduler(optimizer, hps):
    # grab maximum steps
    max_steps = hps.train.num_train_iters

    def cosine_lr(current_step):
        _lr = max(0.0, np.cos((7 * np.pi * current_step) / (16 * max_steps)))
        return _lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_lr)


class TBLog:
    """
    Construc tensorboard writer (self.writer).
    The tensorboard is saved at os.path.join(tb_dir, file_name).
    """

    def __init__(self, tb_dir, file_name):
        self.tb_dir = tb_dir
        self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))

    def update(self, tb_dict, it, suffix=None):
        """
        Args
            tb_dict: contains scalar values for updating tensorboard
            it: contains information of iteration (int).
            suffix: If not None, the update key has the suffix.
        """
        if suffix is None:
            suffix = ''

        for key, value in tb_dict.items():
            self.writer.add_scalar(suffix + key, value, it)


def net_builder(net_params: dict):
    """
    Return ResNet building instance.
    """
    import wrn as net

    # grab the builder class
    builder = net.build_WideResNet()

    # parse the input params
    print("Building ResNet...")
    for key in net_params:
        if hasattr(builder, key):
            print("Setting ", key, ": ", net_params[key])
            setattr(builder, key, net_params[key])

    return builder.build


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def split_ssl_data(data, target, num_labels, num_classes, index=None, include_lb_to_ulb=True):
    """
    data & target is split into labeled and unlabelled data.

    Args
        index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabelled data
    """
    data, target = np.array(data), np.array(target)
    lb_data, lbs, lb_idx = sample_labeled_data(data, target, num_labels, num_classes, index)
    ulb_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx))))  # unlabeled_data index of data
    if include_lb_to_ulb:
        return lb_data, lbs, data, target
    else:
        return lb_data, lbs, data[ulb_idx], target[ulb_idx]


def sample_labeled_data(data, target,
                        num_labels,
                        num_classes,
                        index=None):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    assert num_labels % num_classes == 0
    if not index is None:
        index = np.array(index, dtype=np.int32)
        return data[index], target[index], index

    samples_per_class = int(num_labels / num_classes)

    lb_data = []
    lbs = []
    lb_idx = []
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        idx = np.random.choice(idx, samples_per_class, False)
        lb_idx.extend(idx)

        lb_data.extend(data[idx])
        lbs.extend(target[idx])
    return np.array(lb_data), np.array(lbs), np.array(lb_idx)