import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam

from load_models import NoisyActivation, get_model
from utils import (get_data_cifar, get_data_cifar100, get_data_UTKFace, test_noise_model, visualize)

torch.manual_seed(0)


def run(args):
    # handle arguments
    print(args)
    upper_bound = args['upper_bound']
    batch_size = args['batch_size']
    num_classes = args['num_classes']
    model_path_template = args['model_path_template']
    noise_path_template = args['noise_path_template']
    img_path_template = args['img_path_template']
    model_indexs = args['model_indexs']
    alpha = args['alpha']

    if args['dataset'] == 'UTKFace':
        train_loader, test_loader = get_data_UTKFace(batch_size)
    elif args['dataset'] == 'cifar':
        train_loader, test_loader = get_data_cifar(batch_size)
    elif args['dataset'] == 'cifar100':
        train_loader, test_loader = get_data_cifar100(batch_size)
    else:
        print('ERROR: No dataset named %s.' % args['dataset'])
        exit(-1)

    try:
        dropped_model_indexs = args['dropped_model_indexs']
    except KeyError:
        dropped_model_indexs = []

    try:
        random_dropped = args['random_dropped']
    except KeyError:
        random_dropped = 0

    # train together
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shape = [3, 32, 32]
    num_epoches = 20

    noise_generator = NoisyActivation(shape, device, upper_bound, 1.0).to(device)
    noise_generator.train()
    optimizer = Adam(noise_generator.parameters(), lr=0.01)
    accuracy_criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epoches):
        show_img = True
        for i, data in enumerate(train_loader):
            # step 1: load one batch data
            img, label = data
            img, label = img.to(device), label.to(device)

            if random_dropped:
                random_dropped_indexs = np.random.randint(0, 10, 2)
            else:
                random_dropped_indexs = []

            for model_index in model_indexs:
                if model_index in dropped_model_indexs or model_index in random_dropped_indexs:
                    continue
                # step 2: load model
                model, model_name = get_model(model_index, num_classes)
                model = model.to(device)
                model_path = model_path_template % model_name
                model.load_state_dict(torch.load(model_path))
                model.eval()

                # step 3: train the noise generator on one batch
                running_loss = 0.0
                running_acc = 0.0
                total_train = 0

                noisy_img = noise_generator(img)
                out = model(noisy_img)
                if show_img:
                    show_img = False
                    visualize(img, noisy_img, os.path.join(
                        img_path_template, 'cifar10_%d.png' % epoch))

                accuracy_loss = accuracy_criterion(out, label)
                generate_loss = noise_generator.inner_loss()

                loss = accuracy_loss + alpha * generate_loss

                running_loss += loss.item() * label.size(0)
                total_train += label.size(0)
                _, pred = torch.max(out, 1)
                running_acc += (pred == label).sum().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                del model

                if i % 500 == 0:
                    print('Test model %s %d epoch %d iteration, Loss: %.6f, Acc: %.6f, Accuracy Loss: %.6f Generate '
                          'loss: %.6f' % (model_name, epoch, i, running_loss / total_train, running_acc /
                                          total_train, accuracy_loss.item(), generate_loss.item()))

        for model_index in model_indexs:
            # step 4: load model
            model, model_name = get_model(model_index, num_classes)
            model = model.to(device)
            model_path = model_path_template % model_name
            model.load_state_dict(torch.load(model_path))
            model.eval()

            test_noise_model(model, model_name, noise_generator, test_loader, device)

            del model

        torch.save(noise_generator.state_dict(), noise_path_template % epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Together Models.')
    parser.add_argument('-c', '--config', required=True, type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        args = yaml.safe_load(f.read())
    run(args)
