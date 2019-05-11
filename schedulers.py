from collections import Counter
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, target_lr, gamma=0.1, warmup_epochs=5, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.target_lr = target_lr
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr + (self.target_lr - base_lr) *
                    self.last_epoch / (self.warmup_epochs - 1)
                    for base_lr in self.base_lrs]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]


class WarmupPolynomialLR(_LRScheduler):
    def __init__(self, optimizer, epochs, target_lr, p=2, warmup_epochs=5, last_epoch=-1):
        self.epochs = epochs
        self.target_lr = target_lr
        self.p = p
        self.warmup_epochs = warmup_epochs
        super(WarmupPolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr + (self.target_lr - base_lr) *
                    self.last_epoch / (self.warmup_epochs - 1)
                    for base_lr in self.base_lrs]
        return [self.target_lr * (1 - self.last_epoch / self.epochs)**self.p
                for base_lr in self.base_lrs]


if __name__ == '__main__':
    v = torch.zeros(10)
    optim = torch.optim.SGD([v], lr=0.1)
    scheduler = WarmupMultiStepLR(optim, milestones=[10, 20], target_lr=0.1*64)

    for epoch in range(1, 20+1):
        scheduler.step()

        print(epoch, optim.param_groups[0]['lr'])

    v = torch.zeros(10)
    optim = torch.optim.SGD([v], lr=0.1)
    scheduler = WarmupPolynomialLR(optim, epochs=200, target_lr=0.1*64)

    for epoch in range(1, 200+1):
        scheduler.step()

        print(epoch, optim.param_groups[0]['lr'])
