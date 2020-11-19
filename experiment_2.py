from __future__ import print_function, division
import argparse
import sys

import torch
from torch.utils.data import DataLoader
import torchvision
import pandas as pd

from utils import net_builder
from ssl_dataset import SSL_Dataset, construct_transforms, get_transform
from haparams import create_hparams


def main(args, hps, use_transform=False):
    # construct transforms
    # grab mean and std
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    # overwrite batch size if needed
    batch_size = args.batch_size if args.batch_size else hps.train.eval_batch

    # settings
    settings = ['tr', 'base', 'noise', 'contr']
    path = 'D:/KTH/Deep_learning_adv/Project/fully_trained/kth_project'

    # loop through settings
    full_data = {}
    for setting in settings:

        args.translate = False
        args.noise = False
        args.contrast = False

        if setting == 'tr':
            args.translate = True
        if setting == 'noise':
            args.noise = True
        if setting == 'contr':
            args.contrast = True

        # horizontal flip + crop is on by default
        transform_list = construct_transforms(mean, std, args)
        local_transform = get_transform(mean, std, transform_list, train=True)

        # store accuracies:
        accs = {}

        # loop through seeds
        for i in range(4):
            if args.best:
                checkpoint_path = path + "/" + "saved_models_seed_" + str(
                    i) + "_" + setting + "/cifar10_40/" + "model_best.pth"
                print("Using best model")
            elif args.last:
                checkpoint_path = path + "/" + "saved_models_seed_" + str(
                    i) + "_" + setting + "/cifar10_40/" + "latest_model.pth"
                print("Using latest model")
            else:
                sys.exit("Model type not specified")

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
                                    num_classes=hps.data.num_classes, data_dir=hps.data.data_dir, args=args).get_dset_clean()

            eval_loader = DataLoader(eval_dset, batch_size=batch_size, shuffle=False,
                                     num_workers=1, pin_memory=True)

            acc = 0.0
            with torch.no_grad():
                for images, target in eval_loader:

                    images = images.type(torch.FloatTensor).to(device)
                    # loop over batch
                    if use_transform:
                        image = []
                        for j in range(images.shape[0]):
                            aug_image = torchvision.transforms.functional.to_pil_image(images[j].cpu())
                            aug_image = local_transform(aug_image)
                            image.append(aug_image)
                        images = torch.stack(image, dim=0).type(torch.FloatTensor).to(device)
                    logit = resnet(images)

                    acc += logit.cpu().max(1)[1].eq(target).sum().numpy()

            key = 'seed ' + str(i)
            accs[key] = (1 - acc / len(eval_dset)) * 100

        meann = 0
        for key in accs:
            meann += accs[key]
        meann = meann / 4

        var = 0
        for key in accs:
            var += (accs[key] - meann) ** 2

        var = var / 4

        accs['Mean and variance'] = str(meann) + "+-" + str(var)

        if setting == 'tr':
            key = "Setting: Baseline"
        if setting == 'noise':
            key = "Setting: Noise"
        if setting == 'contr':
            key = "Setting: Contrast"
        if setting == 'base':
            key = "Setting: Removed Translate"

        full_data[key] = accs

    print()
    print()
    for key in full_data:
        print(key)
        acc = full_data[key]
        df = pd.DataFrame.from_dict(acc, orient='index')
        print(df)
        print("-------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # use the best model
    parser.add_argument('--best', action='store_true', help='Use the best model for each setting')

    # use the latest model
    parser.add_argument('--last', action='store_true', help='Use the latest model for each setting')

    # use this to switch between train/eval models
    parser.add_argument('--use_train_model', action='store_true')

    # Experimental arguments
    parser.add_argument('-tr', '--translate', action='store_true', help='Add translation transform during test')
    parser.add_argument('-n', '--noise', action='store_true', help='Add noise transform transform during test')
    parser.add_argument('-c', '--contrast', action='store_true', help='Add contrast transform to weak augment')

    # use this to overwrite the config file
    parser.add_argument('--batch_size', type=int)

    # get the config
    parser.add_argument('-hp', '--hparams', type=str,
                        required=True, help='path to model parameters')

    args = parser.parse_args()
    hps = create_hparams(args.hparams)

    main(args, hps, use_transform=True)
