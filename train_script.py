'''
train_script.py trains a VAE model on the CDRH3 dataset
'''
import torch
from torch.utils.data import DataLoader
from torch import optim

# load our dataset class that handels loading, encoding and accessing
from Datasets import CDRH3MotifDataset as CDRH3
# load our VAE model and criterion
from Models import VaeCdrh3 as Vae, VaeCriterion


DATA_PATH = 'hackathon.csv'
EPOCHS = 10_000
BATCH_SIZE = 1000
LATENT_N = 10
DEVICE = 'cuda:0'  # change to 'cpu' if training on cpu

dataset = CDRH3(DATA_PATH, device=DEVICE)
# crate iterator with training data
data = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                  drop_last=True, prefetch_factor=2)

vae = Vae(latent_n=LATENT_N).to(DEVICE)  # push model onto DEVICE
# initilize optimizer with parameters
opt = optim.Adam(vae.parameters(), weight_decay=0.0001, lr=0.0001)
criterion = VaeCriterion(BATCH_SIZE, len(dataset)).to(DEVICE)

error_min = float('inf')
for epoch in range(EPOCHS):
    print('epoch: ', epoch, flush=True)
    error = 0
    for i, (data_point, label, _) in enumerate(data):  # iterate over dataset
        opt.zero_grad()  # set gradient memory to zero
        predict = vae(data_point)  # one forwardpass for current batch
        error = criterion(predict, label)  # callculate error
        # callculate gradient for all parameters with backwardpass
        error.backward()
        opt.step()  # update step for all parameters
        if i % 10_000 == 0:
            print('within epoch: ', epoch, 'error: ', error, flush=True)
            if error_min > error:
                torch.save(vae.state_dict(),
                           f'vaemodel_epoch{epoch}_iter{i}_error{error}.pt')
                error_min = error
