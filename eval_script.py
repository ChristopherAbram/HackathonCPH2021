'''
example of how to retrieve latent space, dimensionality reduction,
visualization and clustering
'''
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ads
from sklearn.metrics import normalized_mutual_info_score as nis
from sklearn.metrics import mutual_info_score as mis

# load our dataset class that handels loading, encoding and accessing
from Datasets import CDRH3MotifDataset as CDRH3
# load our VAE model and criterion
from Models import VaeCdrh3 as Vae


DATA_PATH = 'hackathon.csv'
MODEL_PATH = 'vaemodel_epoch298_iter0_error5332.66064453125.pt'
LATENT_N = 10
DEVICE = 'cuda:0'  # change to 'cpu' if training on cpu

dataset = CDRH3(DATA_PATH, device=DEVICE)
# crate iterator with data
data = DataLoader(dataset, batch_size=190_000, shuffle=True, num_workers=0,
                  drop_last=False, prefetch_factor=2)

vae = Vae(latent_n=LATENT_N).to(DEVICE)  # push model onto DEVICE
vae.load_state_dict(torch.load(MODEL_PATH))

# get latent space from
with torch.no_grad():
    data_point, label, eval_data = next(iter(data))
    predict = vae(data_point)
    latent = predict['mu']
    ag = eval_data[2]

ag_set = set()
for element in ag:
    ag_set |= {element}
ag_map = {name: i for i, name in enumerate(ag_set)}
ag_color = [ag_map[s] for s in ag]


# VISUALIZATION
# pca dimension reduction
pca = PCA(n_components=2)
x = pca.fit_transform(latent.to('cpu').numpy())


# visualization everything
plt.figure()
plt.scatter(x[:, 0], x[:, 1], c=np.array(ag_color),
            cmap=plt.get_cmap('tab10'), alpha=0.5)
plt.savefig('tab10.png')


# visualize one vs one
plt.figure()
ag2 = [ag_map[i] for i in ag if i in {'Ag12', 'Ag130'}]
idx_x2 = [idx for idx, i in enumerate(ag) if i in {'Ag12', 'Ag130'}]
x2 = np.take(x, idx_x2, axis=0)

plt.scatter(x2[:, 0], x2[:, 1], c=np.array(ag2), alpha=0.5)
plt.savefig('test2.png')

# visualize one vs on for all antigens

# as matrix
plt.figure()
fig, axs = plt.subplots(2, 5)
for number, ag_element in enumerate(ag_set):
    if number < 5:
        idx = [idx for idx, i in enumerate(ag) if i != ag_element]
        x2 = np.take(x, idx, axis=0)
        axs[0, number].scatter(x2[:, 0], x2[:, 1], c='gainsboro', alpha=1)
        idx = [idx for idx, i in enumerate(ag) if i == ag_element]
        x2 = np.take(x, idx, axis=0)
        axs[0, number].scatter(x2[:, 0], x2[:, 1], c='black', alpha=0.5)
    else:
        idx = [idx for idx, i in enumerate(ag) if i != ag_element]
        x2 = np.take(x, idx, axis=0)
        axs[1, number-5].scatter(x2[:, 0], x2[:, 1], c='gainsboro', alpha=1)
        idx = [idx for idx, i in enumerate(ag) if i == ag_element]
        x2 = np.take(x, idx, axis=0)
        axs[1, number-5].scatter(x2[:, 0], x2[:, 1], c='black', alpha=0.5)
fig.savefig('test4.png')

# single images
for ag_element in ag_set:
    plt.figure()
    idx = [idx for idx, i in enumerate(ag) if i != ag_element]
    x2 = np.take(x, idx, axis=0)
    plt.scatter(x2[:, 0], x2[:, 1], c='gainsboro', alpha=1)

    idx = [idx for idx, i in enumerate(ag) if i == ag_element]
    x2 = np.take(x, idx, axis=0)
    plt.scatter(x2[:, 0], x2[:, 1], c='black', alpha=0.5)
    plt.savefig(ag_element + '.png')


# clustering
kmeans = KMeans(n_clusters=10).fit(latent.to('cpu').numpy())
# eval clustering
ads(np.array(ag_color), kmeans.labels_)
nis(np.array(ag_color), kmeans.labels_)

for i in zip(np.array(ag_color), kmeans.labels_):
    print(i)
