import matplotlib.pyplot as plt
import argparse
import re
import sys


# loop through log files
def main(args):
    if args.compact:
        fig, ax1 = plt.subplots()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_xlabel('iters $\cdot 10^4$')
    ax1.set_ylabel('Loss')

    if args.compact:
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.tick_params(axis='y', labelcolor='red')

    else:
        ax2.set_title('Evaluation accuracy')
        ax1.set_title('Training loss')

    ax2.set_xlabel('iters $\cdot 10^4$')
    ax2.set_ylabel('Accuracy')

    if args.type == 'base':
        title = 'Removed translate'
    elif args.type == 'contr':
        title = 'Contrast'
    elif args.type == 'tr':
        title = 'Baseline'
    elif args.type == 'noise':
        title = 'Gaussian noise'
    else:
        sys.exit("Bad argument --type. Available types: [base, tr, contr, noise]")

    styles = ['-', ':', '-.', '--']
    if args.compact:
        color_loss = ['crimson', 'firebrick', 'tomato', 'red']
        color_acc = ['navy', 'royalblue', 'blue', 'dodgerblue']

    # loop through seeds
    for i in range(4):
        path = args.load_path + "/saved_models_seed_" + str(i) + "_" + args.type
        full_path = path + "/cifar10_40/log.txt"

        with open(full_path, 'r') as in_f:
            lines = in_f.readlines()
            iters = []
            accs = []
            losses = []

            for line in lines:

                # use regex to find iterations
                iter = re.search('] (.+?) iteration', line)

                if iter and int(iter.group(1)) <= 340000:
                    iters.append(int(iter.group(1))/10000)

                # use regex to find acc
                acc = re.search("top-1-acc': tensor(.+?),", line)
                if acc:
                    accs.append(float(acc.group(1)[1:]))

                # use regex to find loss
                loss = re.search("total_loss': tensor(.+?),", line)
                if loss:
                    losses.append(float(loss.group(1)[1:]))

            losses = losses[:len(iters)]
            accs = accs[:len(iters)]
            assert (len(iters) == len(losses) == len(accs))
            label = 'seed ' + str(i)

            if args.compact:
                ax1.plot(iters, losses, label=label, linestyle=styles[i], color=color_loss[i])
                ax2.plot(iters, accs, label=label, linestyle=styles[i], color=color_acc[i])
            else:
                ax1.plot(iters, losses, label=label, linestyle=styles[i])
                ax2.plot(iters, accs, label=label, linestyle=styles[i])

    ax1.legend()
    ax2.legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str, default='./results')

    parser.add_argument('--compact', action='store_true', help='Combine loss and accuracy plots')

    # use this to switch between train/eval models
    parser.add_argument('--type', type=str, required=True, help='model type, base/contr/tr/noise')

    args = parser.parse_args()

    main(args)
