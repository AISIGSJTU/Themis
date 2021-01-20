import os
import yaml
import argparse

import torch
import torch.nn as nn
from torch import optim

from load_models import get_model
from utils import get_data_UTKFace, get_data_cifar, get_data_cifar100, train_plain_model, test_plain_model


def run(args):
    # handle arguments
    print(args)
    batch_size = args['batch_size']
    num_classes = args['num_classes']
    model_path_template = args['model_path_template']
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


    torch.manual_seed(0)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epoches = 40

    for model_index in model_indexs:

        max_accuracy = 0.0
        model, model_name = get_model(model_index, num_classes)
        model = model.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)

        for epoch in range(num_epoches):
            train_plain_model(model, train_loader, criterion, optimizer, epoch, DEVICE)
            accuracy = test_plain_model(model, model_name, test_loader, DEVICE)

            if accuracy > max_accuracy:
                torch.save(model.state_dict(), model_path_template % model_name)
                max_accuracy = accuracy
            scheduler.step()

        del model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Plain Models.')
    parser.add_argument('-c', '--config', required=True, type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        args = yaml.safe_load(f.read())
    run(args)
