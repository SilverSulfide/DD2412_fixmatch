import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import os

from utils import init_optimiser, init_lr_scheduler


class FixMatch:
    def __init__(self, net_builder, hps, tb_log=None, logger=None):
        super(FixMatch, self).__init__()

        # momentum update param
        self.num_classes = hps.data.num_classes
        self.ema_m = hps.train.ema_m

        # create the encoders

        self.train_model = net_builder(num_classes=self.num_classes)
        self.eval_model = net_builder(num_classes=self.num_classes)
        self.num_eval_iter = hps.train.log_interval
        self.lambda_u = hps.train.ulb_loss_ratio
        self.tb_log = tb_log
        self.use_hard_label = hps.train.hard_label
        self.label_threshold = hps.train.label_threshold

        # amount of labelled samples in a single forward call
        self.lb = 0

        # set optimiser, scheduler and scaler
        self.optimizer = init_optimiser(self.train_model, hps)
        self.scheduler = init_lr_scheduler(self.optimizer, hps)
        self.scaler = GradScaler()

        self.it = 0

        self.logger = logger
        self.print_fn = logger.info

        # FIXME: get rid of this
        for param_q, param_k in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize the evaluation net
            param_k.requires_grad = False  # do not update by gradient for evaluation net

        self.eval_model.eval()

    def forward(self, inputs):
        """

        :param inputs: concatenated tensor of labelled and unlabelled images
        :return: supervised/unsupervised logits
        """
        # perform inference on the whole input
        logits = self.train_model(inputs)  # concat_batch_size x channels x width x height

        # extract labelled logits for supervised loss
        lb_logits = logits[:self.lb]

        # extract unlabelled logits for unsupervised loss
        ulb_w_logits, ulb_s_logits = logits[self.lb:].chunk(2)

        # free memory
        del logits

        return lb_logits, ulb_w_logits, ulb_s_logits

    def init_lb(self, x):
        """
        :param x: batch of labelled samples

        Initatates the split value that used in forward call
        """
        self.lb = x.shape[0]

    def update_iter(self):
        self.it = self.it + 1

    def loss(self, lb_logits, lb_targets, ulb_w_logits, ulb_s_logits):
        """

        :param lb_logits: supervised logits
        :param lb_targets: supervised targets
        :param ulb_w_logits: weakly-augmented unsupervised logits
        :param ulb_s_logits: strongly-augmented supervised logits
        :return: tuple: supervised, unsupervised loss, combine loss

        Calculates both supervised and unsupervised loss
        """

        # calculate supervised cross entropy loss for the whole batch
        sup_loss = F.cross_entropy(lb_logits, lb_targets, reduction='mean')

        # calculate the unsupervised cross entropy loss

        # get pseudo-label from weakly-supervised logits
        pseudo_labels = F.softmax(ulb_w_logits, dim=1)  # batch x classes

        # get max, compare to the cut-off value
        pseudo_values, pseudo_labels = torch.max(pseudo_labels, dim=1)  # batch x classes
        hard_labels = pseudo_values > self.label_threshold  # bool

        # cross-entropy, ignoring the inputs that do not have a hard label
        unsup_loss = F.cross_entropy(ulb_s_logits, pseudo_labels,
                                     reduction='mean') * hard_labels.float()  # bool ->zeros

        total_loss = sup_loss + self.lambda_u * unsup_loss.mean()

        return sup_loss, unsup_loss.mean(), total_loss

    def backward(self, loss):
        """
        Performs the backward pass using mixed precision.
        https://pytorch.org/docs/stable/amp.html
        """
        # enable mixed precision training
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()


        # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        self.scheduler.step()
        self.train_model.zero_grad()

    # taken from: https://github.com/LeeDoYup/FixMatch-pytorch
    @torch.no_grad()
    def _eval_model_update(self):
        """
        Momentum update of evaluation model (exponential moving average)
        """
        for param_train, param_eval in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_eval.copy_(param_eval * self.ema_m + param_train.detach() * (1 - self.ema_m))

        for buffer_train, buffer_eval in zip(self.train_model.buffers(), self.eval_model.buffers()):
            buffer_eval.copy_(buffer_train)

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        torch.save({'train_model': train_model.state_dict(),
                    'eval_model': eval_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it}, save_filename)

        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model

        for key in checkpoint.keys():
            if hasattr(self, key) and getattr(self, key) is not None:
                if 'train_model' in key:
                    train_model.load_state_dict(checkpoint[key])
                elif 'eval_model' in key:
                    eval_model.load_state_dict(checkpoint[key])
                elif key == 'it':
                    self.it = checkpoint[key]
                else:
                    getattr(self, key).load_state_dict(checkpoint[key])
                self.print_fn(f"Check Point Loading: {key} is LOADED")
            else:
                self.print_fn(f"Check Point Loading: {key} is **NOT** LOADED")
