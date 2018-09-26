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
        TensorDataset(torch.rand(2, 3, 224, 224) * 2 - 1,
                      torch.zeros(2, dtype=torch.long)),
        TensorDataset(torch.randn(2, 3, 224, 224),
                      torch.ones(2, dtype=torch.long))
    ])

    trainer.train(dataset, batch_size=2, n_epochs=1, seed=117)
