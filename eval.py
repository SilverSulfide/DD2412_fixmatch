from __future__ import print_function, division
import os
import argparse

import torch
from torch.utils.data import DataLoader

from utils import net_builder
from ssl_dataset import SSL_Dataset
from haparams import create_hparams


def main(args, hps):
    # overwrite batch size if needed
    batch_size = args.batch_size if args.batch_size else hps.train.eval_batch

    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['train_model'] if args.use_train_model else checkpoint['eval_model']

    resnet_builder = net_builder({'depth': hps.model.depth,
                          'widen_factor': hps.model.widen_factor,
                          'leaky_slope': hps.model.leaky_slope,
                          'dropRate': hps.model.dropout})

    resnet = resnet_builder(num_classes=hps.data.num_classes)
    resnet.load_state_dict(load_model)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    resnet.to(device)
    resnet.eval()

    eval_dset = SSL_Dataset(name=hps.data.dataset, train=False,
                            num_classes=hps.data.num_classes, data_dir=hps.data.data_dir, args=args).get_dset()

    eval_loader = DataLoader(eval_dset, batch_size=batch_size, shuffle=False,
                             num_workers=1, pin_memory=True)

    acc = 0.0
    with torch.no_grad():
        for image, target in eval_loader:
            image = image.type(torch.FloatTensor).to(device)
            logit = resnet(image)

            acc += logit.cpu().max(1)[1].eq(target).sum().numpy()

    print(f"Test Accuracy: {acc / len(eval_dset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str, default='./saved_models/fixmatch/model_best.pth')

    # use this to switch between train/eval models
    parser.add_argument('--use_train_model', action='store_true')

    # use this to overwrite the config file
    parser.add_argument('--batch_size', type=int)

    # get the config
    parser.add_argument('-hp', '--hparams', type=str,
                        required=True, help='path to model parameters')
    # Experimental arguments
    parser.add_argument('-tr', '--translate', action='store_true', help='Add translation transform during test')
    parser.add_argument('-n', '--noise', action='store_true', help='Add noise transform transform during test')
    parser.add_argument('-c', '--contrast', action='store_true', help='Add contrast transform to weak augment')

    args = parser.parse_args()
    hps = create_hparams(args.hparams)

    main(args, hps)
