import os
import argparse
import csv

import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader

from classification import (
    train,
    inference,
    header)
from classification.measure import MeasureR
from classification.customloader import COVID_Dataset


def setup_cuda():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()


def get_default_models():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))


def setup_cli(model_names):
    print(model_names)
    parser = argparse.ArgumentParser(
        description='Train and Test patch based model')
    parser.add_argument(
        '--experimentID', default='%s_20200407_patched_%s', type=str, metavar='E_ID',
        help='ID of Current experiment')
    parser.add_argument('--cv', default='cv5', type=str, metavar='CV_ID',
                        help='Cross Validation Fold')
    parser.add_argument(
        '-d', '--data',
        default='./data_preprocess/standard_data_patched_0922_crossentropy/exp_%s_list_%s.pkl',
        type=str)
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19_bn',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('-k', '--patches', default=50, type=int, metavar='K',
                        help='number of patches for inference')

    parser.add_argument('-c', '--checkpoint', default='./patched_results/checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('-ck_n', '--checkpoint_saved_n', default=2, type=int, metavar='saved_N',
                        help='each N epoch to save model')

    parser.add_argument('--crop_size', type=int, default=224,
                        help='224 then crop to this size')

    # Test Outputs
    parser.add_argument('--test', default=False, dest='test', action='store_true',
                        help='evaluate model on test set')
    parser.add_argument('--results', default='./patched_results/results', type=str, metavar='PATH',
                        help='path to save experiment results (default: results)')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='saved model ID for loading checkpoint (default: none)')

    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    parser.add_argument('--in_memory', default=False, dest='in_memory', action='store_true',
                        help='Load images from /dev/shm')
    return parser.parse_args()


def create_dir(path):
    # Make directory to save
    if not os.path.exists(path):
        os.makedirs(path)


def map_options(args):
    opts = Settings()
    for k, v in args:
        pass


def get_data_loaders(opts):
    train_dataset = COVID_Dataset(
        (opts.crop_size, opts.crop_size), n_channels=3, n_classes=4, mode='train', opts=opts)

    val_dataset = COVID_Dataset(
        (opts.crop_size, opts.crop_size), n_channels=3, n_classes=4, mode='val', opts=opts)

    test_dataset = COVID_Dataset(
        (opts.crop_size, opts.crop_size), n_channels=3, n_classes=4, mode='test', opts=opts)

    image_datasets = {'train': train_dataset,
                      'val': val_dataset,
                      'test': test_dataset}

    batch_size = {'train': header.train_batch_size,
                  'val': header.val_batch_size,
                  'test': header.test_batch_size}
    return {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size[x], num_workers=opts.workers, pin_memory=True)
        for x in ['train', 'val', 'test']}


def main():
    model_names = get_default_models()
    args = setup_cli(model_names)

    experimentID = args.experimentID % (args.arch, args.cv)
    args.data = args.data % ('%s', args.cv)
    print(args.data)
    args.checkpoint_dir = os.path.join(args.checkpoint, experimentID)
    # create checkpoint directory
    create_dir(args.checkpoint_dir)
    create_dir(args.results)
    # args.in_memory = True

    if args.test:
        dataloaders_dict = get_data_loaders(args)
        for phase, dataloader in dataloaders_dict.items():
            plot_file = 'cf_%s_%s_%s.png' % (
                args.arch, phase, args.cv)
            args.cf_plot = os.path.join(args.results, plot_file)
            test_loss, test_acc, pred_d, real_d = inference.main(
                args, dataloader)
            detail_file = 'result_detail_%s_%s_%s.csv' % (
                args.arch, phase, args.cv)
            with open(os.path.join(args.results, detail_file), 'w') as f:
                csv_writer = csv.writer(f)
                for i in range(len(real_d)):
                    x = np.zeros(len(pred_d[i]))
                    x[real_d[i]] = 1
#                  y = np.exp(pred_d[i])/np.sum(np.exp(pred_d[i]))
                    csv_writer.writerow(list(np.array(pred_d[i])) + list(x))

            meaure_file = 'measure_detail_%s_%s_%s.csv' % (
                args.arch, phase, args.cv)
            mr = MeasureR(args.results, test_loss, test_acc,
                          infile=detail_file, outfile=meaure_file)
            mr.output()
            print(' Test Loss:  %.8f, Test Acc:  %.4f' % (test_loss, test_acc))

    else:
        plot_file = 'f1_%s_train_%s.png' % (args.arch, args.cv)
        args.train_plot = os.path.join(args.checkpoint_dir, plot_file)
        train.main(args)


if __name__ == '__main__':
    main()
