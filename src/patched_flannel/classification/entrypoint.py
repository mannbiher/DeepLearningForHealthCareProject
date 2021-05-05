import os
import torchvision.models as models

import train

class Settings(dict):
    """Dict to map args."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def setup_cuda():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()


def get_default_models():
    model_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))


def setup_cli(model_names):
    parser = argparse.ArgumentParser(
        description='Train and Test patch based model')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19_bn',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('-c', '--checkpoint', default='./patched/checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('-ck_n', '--checkpoint_saved_n', default=2, type=int, metavar='saved_N',
                        help='each N epoch to save model')

    parser.add_argument('--crop_size', type=int, default=224,
                        help='224 then crop to this size')

    # Test Outputs
    parser.add_argument('--test', default=False, dest='test', action='store_true',
                        help='evaluate model on test set')
    parser.add_argument('--results', default='./patched/results', type=str, metavar='PATH',
                        help='path to save experiment results (default: results)')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='saved model ID for loading checkpoint (default: none)')

    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    return parser.parse_args()


def create_dir(path):
    # Make directory to save
    if not os.path.exists(path):
        os.makedirs(path)


def map_options(args):
    opts = Settings()
    for k, v in args:
        if k not in ()


def main():
    model_names = get_default_models()
    args = setup_cli(model_names)

    # create checkpoint directory
    create_dir(args.checkpoint)
    create_dir(args.results)

    create_dir(args.)
    train.main()
