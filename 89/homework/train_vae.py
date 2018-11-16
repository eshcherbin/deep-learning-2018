import argparse
import logging
import os

import torch
import torchvision.datasets as datasets
from torch.optim import Adam
from torchvision import transforms

from vae.vae import VAE, loss_function
from vae.trainer import Trainer


def get_config():
    parser = argparse.ArgumentParser(description='Training VAE on CIFAR10')

    parser.add_argument('--log-root', type=str, default='../logs')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--log-name', type=str, default='train_vae.log')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--cifar10', action='store_true', default=False,
                        help='use CIFAR-10 instead of Fashion-MNIST')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train ')
    parser.add_argument('--image-size', type=int, default=28,
                        help='size of images to generate')
    parser.add_argument('--n_show_samples', type=int, default=8)
    parser.add_argument('--show_img_every', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=10)
    config = parser.parse_args()
    config.cuda = not config.no_cuda and torch.cuda.is_available()

    return config


def main():
    config = get_config()
    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_root,
                                             config.log_name)),
            logging.StreamHandler()],
        level=logging.INFO)

    transform = transforms.Compose([transforms.Scale(config.image_size), transforms.ToTensor(),
                                    ])
                                    # transforms.Normalize(0.5, 0.5)])
    dataset = datasets.CIFAR10 if config.cifar10 else datasets.FashionMNIST
    train_set = dataset(train=True, root=config.data_root,
                        download=True, transform=transform)
    test_set = dataset(train=False, root=config.data_root,
                       download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                                               num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=True,
                                              num_workers=4, pin_memory=True)

    vae = VAE(enc_hidden=800, latent_size=40, dec_hidden=800)

    trainer = Trainer(model=vae, train_loader=train_loader, test_loader=test_loader,
                      optimizer=Adam(vae.parameters(), lr=0.0002, betas=(0.5, 0.999)),
                      loss_function=loss_function,
                      device='cuda' if config.cuda else 'cpu')

    for epoch in range(config.epochs):
        trainer.train(epoch, config.log_interval)
        trainer.test(epoch, config.batch_size, config.log_interval,
                     config.show_img_every, config.n_show_samples)


if __name__ == '__main__':
    main()

