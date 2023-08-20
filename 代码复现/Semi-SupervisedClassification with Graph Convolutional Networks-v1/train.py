import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def train(model, data, epoch, optimizer,
          criterion=None, seed=42, fastmode=False):
    loss = 0
    return loss
