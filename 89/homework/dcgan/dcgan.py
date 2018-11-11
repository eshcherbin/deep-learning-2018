import torch.nn as nn


class DCGenerator(nn.Module):

    def __init__(self, image_size, latent_size=100):
        super(DCGenerator, self).__init__()
        self.image_size = image_size
        self.latent_size = latent_size

    def forward(self, data):
        # TODO your code here
        pass


class DCDiscriminator(nn.Module):

    def __init__(self, image_size):
        super(DCDiscriminator, self).__init__()

    def forward(self, data):
        # TODO your code here
        pass
