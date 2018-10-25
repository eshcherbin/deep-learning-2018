import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from timeit import default_timer as timer


class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self,
              dataset,
              batch_size,
              n_epochs,
              val_dataset=None,
              seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        writer = SummaryWriter()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=True)

        start_time = timer()
        for epoch in range(n_epochs):
            epoch_start_time = timer()

            losses = torch.zeros(len(loader))
            for t, (X, y) in enumerate(loader):
                self.optimizer.zero_grad()
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()
                losses[t] = loss

            if val_dataset is not None:
                val_losses = torch.zeros(len(val_loader))
                with torch.no_grad():
                    for t, (X, y) in enumerate(val_loader):
                        y_pred = self.model(X)
                        loss = self.loss_fn(y_pred, y)
                        val_losses[t] = loss

                writer.add_scalars('loss', {'train': losses.mean(),
                                            'val': val_losses.mean()}, epoch)
            else:
                writer.add_scalar('loss', losses.mean(), epoch)
            writer.add_scalar('epoch_time', timer() - epoch_start_time, epoch)

        writer.add_text('finish_time',
                        'Learning finished in {:.3f} seconds'
                        .format(timer() - start_time))
        writer.close()

        return self.model
