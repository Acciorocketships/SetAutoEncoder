import torch
import tqdm
import random
from torch import nn
import torchvision
import torchvision.transforms as transforms
from sae.sae_inner import AutoEncoder
from sae import mse_sparse
import torch_geometric
from matplotlib import pyplot as plt
import numpy as np
import time


def imshow(img, epoch):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.tight_layout()
    plt.savefig(f"/tmp/recon_epoch_{epoch}.png")
    #plt.draw()
    #plt.show(block=False)
    time.sleep(3)

def constrained_sum_sample_pos(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(random.sample(range(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]


class CNNAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
            
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class SeqAE(nn.Module):
    def __init__(self, max_seq_len):
        super().__init__()
        self.cnn_ae = CNNAE()
        self.set_ae = AutoEncoder(
            dim=768, hidden_dim=max_seq_len * 512, max_n=max_seq_len, pe="freq"
        )

    def forward(self, x, b_idx):
        z = self.encode(x, b_idx)
        return self.decode(z)

    def encode(self, x, b_idx):
        # Shape [B*t,f]
        feat = self.cnn_ae.encoder(x).flatten(1)
        z = self.set_ae.encoder(feat, b_idx)
        return z

    def decode(self, z):
        feat, batch = self.set_ae.decoder(z)
        img = self.cnn_ae.decoder(feat.reshape(-1, 48, 4, 4))
        return img, batch



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 256
val_batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='/tmp/datasets', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='/tmp/datasets', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=val_batch_size,
                                         shuffle=False, num_workers=0)
valset = torch.stack([testset[i][0] for i in range(64)], dim=0)

val_seq_lens = torch.tensor([4, 6, 8, 10, 12, 14, 10])
seq_len_range = torch.tensor([4, 16])
num_seqs_range = (1 / (seq_len_range / batch_size)).int().flip(0)
model = SeqAE(seq_len_range[1])
opt = torch.optim.Adam(model.parameters(), lr=0.001)
crit = mse_sparse

# train loop
epochs = 10
pbar = tqdm.trange(epochs, leave=True)
loss = 0
val_loss = 0
for epoch in pbar:
    for data in trainloader:
        inputs, labels = data
        opt.zero_grad()
        num_seqs = random.randint(4, 16)
        seq_lens = torch.tensor(
            constrained_sum_sample_pos(num_seqs, batch_size), dtype=torch.int64
        )
        b_idx = torch.repeat_interleave(torch.arange(seq_lens.numel()), seq_lens)
        recon, recon_b_idx = model(inputs, b_idx)
        # Required for loss
        target_list = []
        target_sum = 0
        for s in seq_lens:
            target_x = inputs[target_sum: target_sum + s]
            target_list.append(torch_geometric.data.Data(x=target_x))
            target_sum += s
        target = torch_geometric.data.Batch.from_data_list(target_list)

        pred_list = []
        pred_sum = 0
        for i in recon_b_idx.unique():
            pred_x = recon[recon_b_idx == i]
            pred_list.append(torch_geometric.data.Data(x=pred_x))
        pred = torch_geometric.data.Batch.from_data_list(pred_list)

        loss = crit(target, pred)
        pbar.set_description(f"train loss: {loss:.3f} val loss: {val_loss:.3f}")
        loss.backward()
        opt.step()

    val_loss = 0
    inputs = valset
    b_idx = torch.repeat_interleave(torch.arange(val_seq_lens.numel()), val_seq_lens)
    with torch.no_grad():
        recon, recon_b_idx = model(inputs, b_idx)
    # Required for loss
    target_list = []
    target_sum = 0
    for s in val_seq_lens:
        target_x = inputs[target_sum: target_sum + s]
        target_list.append(torch_geometric.data.Data(x=target_x))
        target_sum += s
    target = torch_geometric.data.Batch.from_data_list(target_list)

    pred_list = []
    pred_sum = 0
    for i in recon_b_idx.unique():
        pred_x = recon[recon_b_idx == i]
        pred_list.append(torch_geometric.data.Data(x=pred_x))
    pred = torch_geometric.data.Batch.from_data_list(pred_list)

    viz = torchvision.utils.make_grid(
        torch.cat([target.x[:16], pred.x[:16]], dim=0), nrow=16
    )
    imshow(viz, epoch)

    val_loss = crit(target, pred)
    pbar.set_description(f"train loss: {loss:.3f} val loss: {val_loss:.3f}")



