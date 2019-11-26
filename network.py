import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, network_type='fc', latent_dim=20):
        super(VAE, self).__init__()
        self.network_type = network_type
        if network_type == 'fc':
            self.encoder = nn.Sequential(
                Layer(784, 400, process='fc', activation='relu', normalize='none'))
            self.mu = Layer(400, latent_dim, process='fc', activation='none', normalize='none')
            self.log_var = Layer(400, latent_dim, process='fc', activation='none', normalize='none')
            self.decoder = nn.Sequential(
                Layer(latent_dim, 400, process='fc', activation='relu', normalize='none'),
                Layer(400, 784, process='fc', activation='sigmoid', normalize='none'))
            
        elif network_type == 'cnn':
            self.encoder = nn.Sequential(
                Layer(1, 16, process='conv', activation='relu', normalize='none'),
                Layer(16, 32, process='conv', activation='relu', normalize='none'))
            self.mu = Layer(32*7*7, latent_dim, process='fc', activation='none', normalize='none')
            self.log_var = Layer(32*7*7, latent_dim, process='fc', activation='none', normalize='none')
            self.fc = Layer(latent_dim, 32*7*7, process='fc', activation='none', normalize='none')
            self.decoder = nn.Sequential(
                Layer(32, 16, process='deconv', activation='relu', normalize='none'),
                Layer(16, 1, process='deconv', activation='sigmoid', normalize='none'))
            
    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        latent_vec = eps.mul(std).add_(mu)
        return latent_vec
    
    def forward(self, x):
        if self.network_type == 'fc':x=x.view(x.size(0), -1)
        h = self.encoder(x)
        if self.network_type == 'cnn':h=h.view(h.size(0), -1)
        mu, log_var = self.mu(h), self.log_var(h)
        latent_vector = self.reparameterization(mu, log_var)
        if self.network_type == 'cnn':
            latent_vector=self.fc(latent_vector).view(h.size(0), 32, 7, 7)
        tilde_x = self.decoder(latent_vector)
        return tilde_x, mu, log_var
            
            
class Layer(nn.Module):
    def __init__(self, in_features, out_features, process='fc', activation='relu', normalize='none'):
        super(Layer, self).__init__()
        if process == 'fc':
            self.process = nn.Linear(in_features, out_features)
        elif process == 'conv':
            self.process = nn.Conv2d(in_features, out_features,
                                kernel_size=4, stride=2, padding=1)
        elif process == 'deconv':
            self.process = nn.ConvTranspose2d(in_features, out_features,
                                kernel_size=4, stride=2, padding=1)
        elif process == 'none':
            self.process = None
        else:
            assert 1, '%s is not supported.' % process
            
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'none':
            self.act = None
        else:
            assert 1, '%s is not supported.' % activation
            
        if normalize == 'batch':
            self.norm = nn.BatchNorm2d(out_features)
        elif normalize == 'none':
            self.norm = None
        else:
            assert 1, '%s is not supported.' % activation
            
    def forward(self, x):
        if self.process:
            x = self.process(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x
            