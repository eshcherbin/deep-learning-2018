import torch.nn as nn


class DCGenerator(nn.Module):

    def __init__(self, image_size, latent_size=100, image_layers=3):
        super(DCGenerator, self).__init__()
        self.image_size = image_size
        self.latent_size = latent_size

        self.fc = nn.Linear(self.latent_size, (self.image_size // 8)**2 * 512)
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, image_layers, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, data):
        fc = self.fc(data.view(-1, self.latent_size))
        return self.deconv(fc.view(-1,
                                   512,
                                   self.image_size // 8,
                                   self.image_size // 8))


class DCDiscriminator(nn.Module):

    def __init__(self, image_size, image_layers=3):
        super(DCDiscriminator, self).__init__()
        self.image_size = image_size
        self.conv = nn.Sequential(
            nn.Conv2d(image_layers, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Linear((self.image_size // 8)**2 * 512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        conv = self.conv(data)
        return self.sigmoid(
            self.fc(conv.view(-1, (self.image_size // 8)**2 * 512))
        )
