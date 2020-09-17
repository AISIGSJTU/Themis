import argparse
import os

import torch
import torch.nn as nn
import yaml
from torch.optim import Adam

from load_models import get_model, NoisyActivation
from utils import get_data_UTKFace, get_data_cifar, get_data_cifar100, train_noise_model, test_noise_model


def run(args):
    # handle arguments
    print(args)
    upper_bound = args['upper_bound']
    batch_size = args['batch_size']
    num_classes = args['num_classes']
    model_path_template = args['model_path_template']
    noise_path_template = args['noise_path_template']
    model_indexs = args['model_indexs']
    if args['dataset'] == 'UTKFace':
        train_loader, test_loader = get_data_UTKFace(batch_size)
    elif args['dataset'] == 'cifar':
        train_loader, test_loader = get_data_cifar(batch_size)
    elif args['dataset'] == 'cifar100':
        train_loader, test_loader = get_data_cifar100(batch_size)
    else:
        print('ERROR: No dataset named %s.' % args['dataset'])
        exit(-1)

    # train together
    torch.manual_seed(0)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    shape = [3, 32, 32]
    num_epoches = 20
    alpha = 0.2

    noise_generator = NoisyActivation(shape, DEVICE, upper_bound, 1.0).to(DEVICE)
    optimizer = Adam(noise_generator.parameters(), lr=0.01)
    accuracy_criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epoches):
        for model_index in model_indexs:
            model, model_name = get_model(model_index, num_classes)
            model = model.to(DEVICE)
            model_path = model_path_template % model_name
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print('*' * 25, model_name, '*' * 25)

            train_noise_model(model, noise_generator, train_loader, epoch, alpha, accuracy_criterion, optimizer, DEVICE)

            test_noise_model(model, model_name, noise_generator, test_loader, DEVICE)

            del model

        torch.save(noise_generator.state_dict(), noise_path_template % epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Together Models.')
    parser.add_argument('-c', '--config', required=True, type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        args = yaml.safe_load(f.read())
    run(args)
