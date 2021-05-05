def setup_cli():
    parser = argparse.ArgumentParser(description='Train and Test patch based model')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19_bn',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

    parser.add_argument('-c', '--checkpoint', default='./explore_version_03/checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('-ck_n', '--checkpoint_saved_n', default=2, type=int, metavar='saved_N',
                    help='each N epoch to save model')

    # Test Outputs
    parser.add_argument('--test', default = False, dest='test', action='store_true',
                    help='evaluate model on test set')
    parser.add_argument('--results', default='./explore_version_03/results', type=str, metavar='PATH',
                    help='path to save experiment results (default: results)')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='saved model ID for loading checkpoint (default: none)')
                    
                    
    parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
    return parser.parse_args()


def main():
    pass