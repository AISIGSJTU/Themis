import os
import yaml

import torch
import torch.nn as nn
from torch.optim import Adam

from load_models import NoisyActivation, get_model
from utils import get_test_data_cifar, test_noise_model


def run():
    torch.manual_seed(0)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    upper_bound = 1.0
    shape = [3, 32, 32]
    batch_size = 64
    num_classes = 10
    test_loader = get_test_data_cifar()

    for model_index in [0, 1, 7, 8, 9, 12, 18, 21, 32, 34]:
        model, model_name = get_model(model_index, num_classes)
        trained_model_path = './out/CIFAR10/models/trained_cifar10_%s.pt' % model_name
        noise_path = './out/CIFAR10/models/train_noise_epoch_19.pt'

        model = model.to(DEVICE)
        model.load_state_dict(torch.load(trained_model_path))

        noise_generator = NoisyActivation(shape, DEVICE, upper_bound, 1.0).to(DEVICE)
        noise_generator.load_state_dict(torch.load(noise_path))
        test_noise_model(model, model_name, noise_generator, test_loader, DEVICE)
        del model, noise_generator


if __name__ == '__main__':
    run()