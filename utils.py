from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.utils import make_grid


def load_state_dict_from_module(path):
    """
    load state dict
    the pretrained models, like alexnet_best.pth.tar are trained in multiple GPUs,
    so we must to preprocess them to train or evlauate on a single GPU.
    :param path: location of the pretrained models
    :return:
    """
    checkpoint = torch.load(path)
    state_dict = checkpoint['state_dict']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # remove 'module.'
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict


def get_data_UTKFace(batch_size=64, workers=0):
    """
    :param batch_size:
    :param workers:
    :return:
    """
    print('Prepare UTKFace...')
    image_data = torchvision.datasets.ImageFolder(
        root='./data/UTKFace/train',
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1))
        ])
    )

    lengths = [int(len(image_data) * 0.8), len(image_data) - int(len(image_data) * 0.8)]
    train_data, test_data = random_split(image_data, lengths)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=workers)

    print('Finish preparing UTKFace, %d train images and %d test images.' % (len(train_data), len(test_data)))
    return train_loader, test_loader


def get_test_data_UTKFace(batch_size=64, workers=0):
    """
    :param batch_size:
    :param workers:
    :return:
    """
    print('Prepare Test UTKFace...')
    data = torchvision.datasets.ImageFolder(
        root='./data/UTKFace/test',
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1))
        ])
    )

    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=workers)

    print('Finish preparing Test UTKFace, %d test images.' % len(data))
    return loader


def visualize(images, noisy_images, filename):
    """
    :param images:
    :param noisy_images:
    :param filename:
    """
    fig, axs = plt.subplots(2)

    images = make_grid(images[0:8], padding=0).cpu().numpy().transpose([1, 2, 0])
    noisy_images = make_grid(noisy_images[0:8].detach(), padding=0).cpu().numpy().transpose([1, 2, 0])

    img = (images - np.min(images)) / (np.max(images) - np.min(images))
    plt.subplot(2, 1, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

    img = (noisy_images - np.min(noisy_images)) / (np.max(noisy_images) - np.min(noisy_images))
    plt.subplot(2, 1, 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

    plt.savefig(filename)
    plt.close(fig)


def get_data_cifar(batch_size=64, workers=0):
    """
    :param batch_size:
    :param workers:
    :return:
    """
    data = CIFAR10(root='./data/cifar',
                   transform=transforms.Compose([
                       transforms.RandomCrop(32, padding=4),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    #    transforms.Normalize((0., 0., 0.), (1, 1, 1))
                   ]),
                   download=True)

    train_data, test_data = random_split(data, [40000, 10000])

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers
    )
    return train_loader, test_loader


def get_test_data_cifar(batch_size=64, workers=0):
    """
    :param batch_size:
    :param workers:
    :return:
    """
    test_loader = DataLoader(
        CIFAR10(root='./data/cifar', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            # transforms.Normalize((0, 0, 0), (1, 1, 1))
        ]), download=True),
        batch_size=batch_size,
        num_workers=workers,
    )
    return test_loader


def get_data_cifar100(batch_size=64, workers=0):
    """
    :param batch_size:
    :param workers:
    :return:
    """
    data = CIFAR100(root='./data/cifar100',
                    transform=transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ]),
                    download=True)

    train_data, test_data = random_split(data, [40000, 10000])

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers
    )
    return train_loader, test_loader


def get_test_data_cifar100(batch_size=64, workers=0):
    """
    :param batch_size:
    :param workers:
    :return:
    """
    test_loader = DataLoader(
        CIFAR100(root='./data/cifar100', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]), download=True),
        batch_size=batch_size,
        num_workers=workers,
    )
    return test_loader


def train_plain_model(model, train_loader, criterion, optimizer, epoch, device):
    """
    train plain model

    :param model:
    :param train_loader:
    :param criterion:
    :param optimizer:
    :param epoch:
    :param device:
    """
    print('*' * 25, 'epoch %d' % epoch, '*' * 25)

    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_train = 0
    for i, data in enumerate(train_loader):
        img, label = data
        img, label = img.to(device), label.to(device)

        out = model(img)
        loss = criterion(out, label)

        running_loss += loss.item() * label.size(0)
        total_train += label.size(0)

        _, pred = torch.max(out, 1)

        running_acc += (pred == label).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Finish %d epoch, Loss: %.6f, Acc: %.6f' % (epoch, running_loss / total_train, running_acc / total_train))


def test_plain_model(model, model_name, test_loader, device):
    """
    test plain model

    :param model_name:
    :param model:
    :param test_loader:
    :param device:
    :return:
    """
    model.eval()
    eval_acc = 0.0
    total_test = 0
    for data in test_loader:
        img, label = data
        img, label = img.to(device), label.to(device)

        out = model(img)

        total_test += label.size(0)
        _, pred = torch.max(out, 1)
        eval_acc += (pred == label).sum().item()

    print('Test %s Acc: %.6f' % (model_name, eval_acc / total_test))
    return eval_acc / total_test


def train_noise_model(model, noise_generator, train_loader, epoch, alpha, accuracy_criterion, optimizer, DEVICE,
                      train_model=False):
    """
    train noise generator

    :param train_model:
    :param model:
    :param noise_generator:
    :param train_loader:
    :param epoch:
    :param weight: alpha
    :param accuracy_criterion: cross_entropy
    :param optimizer:
    :param DEVICE:
    loss = cross_entropy + alpha * generate_loss
    """
    print('*' * 25, 'epoch %d' % epoch, '*' * 25)
    noise_generator.train()

    if train_model:
        model.train()

    running_loss = 0.0
    running_acc = 0.0
    total_train = 0
    for i, data in enumerate(train_loader):
        img, label = data
        img, label = img.to(DEVICE), label.to(DEVICE)

        noisy_img = noise_generator(img)
        out = model(noisy_img)

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

        if i % 100 == 0:
            print('%d epoch %d iteration, Loss: %.6f, Acc: %.6f, Accuracy Loss: %.6f, '
                  'Generate loss: %.6f' % (
                      epoch, i, running_loss / total_train, running_acc / total_train, accuracy_loss.item(),
                      generate_loss.item()))

    print('Finish %d epoch, Loss: %.6f, Acc: %.6f' % (epoch, running_loss / total_train, running_acc / total_train))


def test_noise_model(model, model_name, noise_generator, test_loader, DEVICE, ):
    """
    test noise generator

    :param model:
    :param model_name:
    :param noise_generator:
    :param test_loader:
    :param DEVICE:
    :return:
    """
    noise_generator.eval()
    model.eval()

    eval_acc = 0.0
    total_test = 0
    for data in test_loader:
        img, label = data
        img, label = img.to(DEVICE), label.to(DEVICE)

        noisy_img = noise_generator(img)
        out = model(noisy_img)

        total_test += label.size(0)
        _, pred = torch.max(out, 1)
        eval_acc += (pred == label).sum().item()

    print('Test %s Acc: %.6f' % (model_name, eval_acc / total_test))
    return eval_acc / total_test
