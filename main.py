import os
import shutil
import argparse
import multiprocessing

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from network import VAE

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=100)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('--display_interval', type=int, default=100)
    parser.add_argument('--network_type', choices=['fc', 'cnn'], type=str, default='fc')
    parser.add_argument('--save_dir', type=str, default='img')
    return parser.parse_args()

def loss_function(tilde_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(tilde_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD, BCE, KLD

def training(epoch, model, opt, train_items, args, tensorboard):
    model.train()
    for idx, (data, _) in enumerate(train_items):
        x = data.cuda()
        opt.zero_grad()
        tilde_x, mu, log_var = model(x)
        total_loss, bce, kl = loss_function(tilde_x, x, mu, log_var)
        total_loss.backward()
        opt.step()
        if idx % args.display_interval == 0:
            print('Training Epoch: {} [{}/{} ({:.0f}%)] | Total loss: {:.6f} | D_KL: {:.6f} | BCE: {:.6f} |'.format(epoch, 
                                                                                                                  idx * len(x), 
                                                                                                                  len(train_items.dataset),
                                                                                                                  100. * idx / len(train_items),
                                                                                                                  total_loss.item(), kl.item(), bce.item()))
            tensorboard.add_scalars('Loss', {'total loss': total_loss.item(),
                                             'D_kl': kl.item(),
                                             'BCE': bce.item()}, global_step=epoch)
            
def test(epoch, test_items, model, args, tensorboard):
    model.eval()
    for idx, (data, _) in enumerate(test_items):
        x = data.cuda()
        with torch.no_grad():
            tilde_x, _, _ = model(x)
        tensorboard.add_images('Image', 
                               tilde_x.view(args.batch_size, 1, 28, 28), 
                               global_step=epoch)
        save_image(tilde_x.view(args.batch_size, 1, 28, 28),
                   os.path.join(args.save_dir, 'samples%d.jpg' % (epoch)))
            
if __name__ == '__main__':
    args = config()
    tensorboard = SummaryWriter(log_dir='logs')
    if os.path.isdir(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)
    
    device = torch.device('cuda: {}'.format(args.gpu))
    model = VAE(network_type=args.network_type, latent_dim=20).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    
    train_items = DataLoader(
        datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True)
    
    test_items = DataLoader(
        datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size,
        shuffle=True)
    
    for epoch in range(1, args.epoch+1):
        training(epoch, model, opt, train_items, args, tensorboard)
        test(epoch, test_items, model, args, tensorboard)
        torch.save(model.state_dict(), 'model')