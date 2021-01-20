import os
import yaml

import torch
import torch.nn as nn
from torch.optim import Adam

from load_models import NoisyActivation, get_model
from utils import get_data_cifar, train_noise_model, test_noise_model


def run():
    torch.manual_seed(0)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    upper_bound = 3.0
    lr = 0.01
    alpha = 0.2

    shape = [3, 32, 32]
    batch_size = 64
    num_epoches = 20
    num_classes = 10
    train_loader, test_loader = get_data_cifar(batch_size)

    for model_index in [0, 1, 7, 8, 9, 12, 18, 21, 32, 34]:
        model, model_name = get_model(model_index, num_classes)
        model_path = './out/CIFAR10/models/cifar10_%s.pt' % model_name
        trained_model_path = './out/CIFAR10/models/trained_cifar10_%s.pt' % model_name
        noise_path = './out/CIFAR10/models/trained_noise_%s.pt' % model_name

        model = model.to(DEVICE)
        model.load_state_dict(torch.load(model_path))

        noise_generator = NoisyActivation(shape, DEVICE, upper_bound, 1.0).to(DEVICE)
        optimizer = Adam([{'params': model.parameters()}, {'params': noise_generator.parameters()}], lr=lr)
        accuracy_criterion = nn.CrossEntropyLoss()

        max_accuracy = 0.0

        for epoch in range(num_epoches):
            train_noise_model(model, noise_generator, train_loader, epoch, alpha,
                            accuracy_criterion, optimizer, DEVICE, train_model=True)

            accuracy = test_noise_model(model, model_name, noise_generator, test_loader, DEVICE)

            if accuracy > max_accuracy:
                torch.save(noise_generator.state_dict(), noise_path)
                torch.save(model.state_dict(), trained_model_path)
                max_accuracy = accuracy

        del model, noise_generator


if __name__ == '__main__':
    run()