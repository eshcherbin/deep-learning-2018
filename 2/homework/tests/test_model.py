from my_resnext import *
from torch.utils.data import TensorDataset, ConcatDataset
import torch.optim as optim
import torch.nn as nn
import torch


def test_random():
    torch.manual_seed(225)

    loss_fn = nn.CrossEntropyLoss()
    model = resnext50()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, optimizer, loss_fn)
    dataset = ConcatDataset([
        TensorDataset(torch.rand(1, 3, 224, 224) * 2 - 1,
                      torch.zeros(1, dtype=torch.long)),
        TensorDataset(torch.randn(1, 3, 224, 224),
                      torch.ones(1, dtype=torch.long))
    ])

    trainer.train(dataset, batch_size=2, n_epochs=1, seed=117)
