import torch
from torch import nn
import torch.nn.functional as F


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
