import argparse
import time

import torch
import yaml

from load_models import get_model
from utils import get_test_data_UTKFace, get_test_data_cifar, get_test_data_cifar100, test_plain_model


def run(args):
    # handle arguments
    print(args)
    batch_size = args['batch_size']
    num_classes = args['num_classes']
    model_path_template = args['model_path_template']
    model_indexs = args['model_indexs']
    if args['dataset'] == 'UTKFace':
        test_loader = get_test_data_UTKFace(batch_size)
    elif args['dataset'] == 'cifar':
        test_loader = get_test_data_cifar(batch_size)
    elif args['dataset'] == 'cifar100':
        test_loader = get_test_data_cifar100(batch_size)
    else:
        print('ERROR: No dataset named %s.' % args['dataset'])
        exit(-1)

    # test plain model
    torch.manual_seed(0)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_index in model_indexs:
        time_start = time.time()

        model, model_name = get_model(model_index, num_classes)
        model = model.to(DEVICE)

        model_path = model_path_template % model_name
        model.load_state_dict(torch.load(model_path))
        test_plain_model(model, model_name, test_loader, DEVICE)

        time_end = time.time()
        print('Time to evaluate plain %s: %.3fs' % (model_name, time_end - time_start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Plain Models.')
    parser.add_argument('-c', '--config', required=True, type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        args = yaml.safe_load(f.read())
    run(args)
