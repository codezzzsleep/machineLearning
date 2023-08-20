import os.path
import time
import argparse
import numpy as np
import torch
from dataset import load_data, load_dataset
from model import MyNet
from train import train
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import utils

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = load_dataset()
data = dataset[0].to(device=device)
model = MyNet(num_feature=dataset.num_features, num_hidden=args.hidden,
              num_classes=dataset.num_classes, dropout=args.dropout).to(device)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
path = utils.create_result_folder()
write = SummaryWriter(os.path.join(path, 'log'))

train(model, data, args.epochs, optimizer, path, write,
      seed=args.seed, fastmode=args.fastmode)
