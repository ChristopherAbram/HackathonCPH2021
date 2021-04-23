import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np


class SampleZ(nn.Module):
    def __init__(self, device=None):
        super(SampleZ, self).__init__()
        self.device = device

    def forward(self, x):
        mu, log_sigma = x
        std = torch.exp(0.5 * log_sigma).to(self.device)
        with torch.no_grad():
            epsilon = torch.randn_like(std).to(self.device)
        return mu + std * epsilon


class VaeCdrh3(nn.Module):
    """VaeCdrh3 implements a VAE model in pyTorch"""
    def __init__(self, latent_n: int, motif_length: int = 11,
                 aa_number: int = 20):
        """
        Args:
            latent_n (int): Size of the latent space of the VAE
            motif_length (int): motif_length of the input
            aa_number (int): number of categories (amino acids)
        """
        super(VaeCdrh3, self).__init__()
        self.device = nn.Parameter(torch.empty(0))
        self.motif_length = motif_length
        self.latent_n = latent_n
        self.aa_number = aa_number

        # encoder:
        self.fc1 = nn.Linear(motif_length * aa_number, 500)
        self.fc2 = nn.Linear(500, 500)
        # create mu and log_var for each latent output
        self.fc3 = nn.Linear(500, latent_n * 2)

        # decoder
        self.fc4 = nn.Linear(latent_n, 500)
        self.fc5 = nn.Linear(500, 500)
        # creat output layer for each letter in motif
        self.out_layers = nn.ModuleList([nn.Linear(500, aa_number)
                                        for i in range(motif_length)])

    def forward(self, x) -> torch.FloatTensor:
        output = dict()
        # flaten input
        x = x.view(-1, self.motif_length * self.aa_number)

        # encoder
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))

        # callculate KL-divergence
        mean = x[:, 0: self.latent_n]
        log_var = x[:, self.latent_n:]
        output['mu'] = mean.detach()
        output['log_var'] = log_var.detach()
        output['KLD'] = 0.5 * torch.sum(mean.pow(2) - log_var
                                        + log_var.exp() - 1, dim=1)

        # reparameterization
        sigma = torch.exp(0.5 * log_var)
        z = mean + sigma * torch.randn(self.latent_n).to(self.device.device)

        # decoder
        x = F.leaky_relu(self.fc4(z))
        x = F.leaky_relu(self.fc5(x))
        # create logsoftmax output for each letter in the motif
        for i, layer in enumerate(self.out_layers):
            output[i] = F.log_softmax(layer(x), dim=1)
        return output


class VaeEmb(nn.Module):
    def __init__(self, layer_config, device):
        super(VaeEmb, self).__init__()
        self.device = device
        self.layer_config = layer_config
        self.encoder_layer_config = self.layer_config[0]
        self.decoder_layer_config = self.layer_config[1]

        # Encoder layers:
        encoder_layers_num = len(self.encoder_layer_config)
        encoder_layers = []
        for i in range(1, encoder_layers_num - 1):
            in_dim, out_dim = self.encoder_layer_config[i - 1], self.encoder_layer_config[i]
            encoder_layers.append(('en_lin_%d' % i, nn.Linear(in_dim, out_dim)))
            encoder_layers.append(('en_lin_batchnorm_%d' % i, nn.BatchNorm1d(out_dim)))
            encoder_layers.append(('en_act_%d' % i, nn.ReLU()))
        self.encoder_ = nn.Sequential(OrderedDict(encoder_layers))

        # Latent space layer (mu & log_var):
        in_dim, out_dim = \
            self.encoder_layer_config[encoder_layers_num - 2], \
            self.encoder_layer_config[encoder_layers_num - 1]
        self.latent_dim = out_dim
        self.en_mu = nn.Linear(in_dim, out_dim)
        self.en_mu_batchnorm = nn.BatchNorm1d(out_dim)
        self.en_log_var = nn.Linear(in_dim, out_dim)
        self.en_log_var_batchnorm = nn.BatchNorm1d(out_dim)

        # Sample from N(0., 1.)
        self.sample = SampleZ(device=device)

        # Decoder layers:
        decoder_layers_num = len(self.decoder_layer_config)
        decoder_layers = []
        for i in range(1, decoder_layers_num - 1):
            in_dim, out_dim = self.decoder_layer_config[i - 1], self.decoder_layer_config[i]
            decoder_layers.append(('de_lin_%d' % i, nn.Linear(in_dim, out_dim)))
            decoder_layers.append(('de_lin_batchnorm_%d' % i, nn.BatchNorm1d(out_dim)))
            decoder_layers.append(('de_act_%d' % i, nn.ReLU()))

        # Last layer of decoder:
        in_dim, out_dim = \
            self.decoder_layer_config[decoder_layers_num - 2], \
            self.decoder_layer_config[decoder_layers_num - 1]
        decoder_layers.append(('de_lin_%d' % (decoder_layers_num - 1), nn.Linear(in_dim, out_dim)))
        decoder_layers.append(('de_act_%d' % (decoder_layers_num - 1), nn.Sigmoid()))
        
        self.decoder = nn.Sequential(OrderedDict(decoder_layers))
        if self.device:
            self.to(self.device)
    

    def encode(self, x):
        x = self.encoder_(x)
        mu = self.en_mu(x)
        mu = self.en_mu_batchnorm(mu)
        log_var = self.en_log_var(x)
        log_var = self.en_log_var_batchnorm(log_var)
        return mu, log_var

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x, _, __ = self._forward(x)
        return x

    def _forward(self, x):
        mu, log_var = self.encode(x)
        z = self.sample([mu, log_var])
        x = self.decode(z)
        return x, mu, log_var

    def loss(self, y_true, y_pred, mu, log_var):
        # E[log P(X|z)]
        recon = torch.sum(F.binary_cross_entropy(y_pred, y_true.data, reduction='none'), dim=1)
        # D_KL(Q(z|X) || P(z|X))
        kld = 0.5 * torch.sum(
            torch.exp(log_var) + torch.square(mu) - 1. - log_var, dim=1)
        return (kld + recon).mean()

    def get_layer_string(self):
        layers = self.layer_config
        layer_string = '-'.join(str(x) for x in layers[0]) + '-' + '-'.join(str(x) for x in np.array(layers[1])[1:])
        return layer_string


class VaeCriterion:
    """VaeCriterion implements NLLLoss for amino acid motifs"""
    def __init__(self, batch_size: int, data_length: int):
        """
        Args:
            batch_size (int): batch_size used during training
            data_length (int): number of datapoints used for training
        """
        self.factor = data_length / batch_size
        self.criterion = torch.nn.NLLLoss(reduction='sum')

    def __call__(self, input_, target) -> torch.Tensor:
        for i in range(target.shape[1]):
            crit = 0
            kl = 0
            crit += self.criterion(input_[i], target[:, i])
            kl += torch.sum(input_['KLD'])
        return self.factor * crit + kl

    def to(self, device: str):
        out = self.__class__(1, 1)
        out.factor = self.factor
        out.criterion = self.criterion.to(device)
        return out
