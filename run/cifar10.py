import os
import torch
import tqdm
import random
from torch import nn
import torchvision
import torchvision.transforms as transforms
from sae.sae_inner import AutoEncoder
from sae import mse_sparse, get_loss_idxs
import torch_geometric
from matplotlib import pyplot as plt
import numpy as np
import time
import wandb



def imshow(img, epoch):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.tight_layout()
    os.makedirs("/tmp/setae", exist_ok=True)
    plt.savefig(f"/tmp/setae/recon_epoch_{epoch:03}.png")

def constrained_sum_sample_pos(n, total, min_num, max_num):
    """Return a randomly chosen list of n positive integers summing to total,
    where each integer is between [min_num, max_num) (not including max_num). Note
    that the final sample may sometimes be less than min_num."""
    nums = []
    running_sum = 0
    while running_sum < total:
        nums += [random.randint(min_num, max_num - 1)]
        running_sum = sum(nums)
    nums[-1] = total - sum(nums[:-1])
    return nums

def constrained_sum_sample_pos2(n, total):
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
            #nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
            #nn.ReLU(),
            
        )
        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            #nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Tanh(),
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
            dim=768, hidden_dim=max_seq_len * 768, max_n=max_seq_len, pe="onehot"
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
        feat, batch, pred_seq_lens = self.set_ae.decoder(z)
        img = self.cnn_ae.decoder(feat.reshape(-1, 48, 4, 4))
        return img, batch, pred_seq_lens


def main():
    device = "cpu"
    wandb.init(project="sae_cifar10")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 256
    val_batch_size = 64

    trainset = torchvision.datasets.CIFAR10(root='/tmp/datasets', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='/tmp/datasets', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=val_batch_size,
                                             shuffle=False, num_workers=0)
    valset = torch.stack([testset[i][0] for i in range(64)], dim=0)

    #val_seq_lens = torch.tensor([4, 6, 8, 10, 12, 14, 10], device=device)
    val_seq_lens = torch.tensor([4] * 16, device=device)
    seq_len_range = torch.tensor([4, 8], device=device)
    num_seqs_range = (1 / (seq_len_range / batch_size)).int().flip(0)
    model = SeqAE(seq_len_range[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    #crit = torch.nn.functional.binary_cross_entropy#mse_sparse
    crit = torch.nn.functional.mse_loss
    #crit = mse_sparse

    # train loop
    epochs = 100
    pbar = tqdm.trange(epochs, position=0, desc="All Epochs", unit="epoch")
    loss = 0
    val_loss = 0
    for epoch in pbar:
        pbar2 = tqdm.tqdm(
            trainloader, position=1, desc=f"Epoch {epoch}", unit="batch", leave=False
        )
        for data in pbar2:
            targets, _ = data
            targets = targets.to(device)
            opt.zero_grad()
            num_seqs = random.randint(*num_seqs_range)
            seq_lens = torch.tensor(
                constrained_sum_sample_pos(num_seqs, targets.shape[0], *seq_len_range),
                dtype=torch.int64, device=device
            )
            b_idx = torch.repeat_interleave(
                torch.arange(seq_lens.numel(), device=device), seq_lens
            )
            recon, recon_b_idx, pred_seq_lens = model(targets, b_idx)
            pred_loss_idx, target_loss_idx = get_loss_idxs(
                pred_seq_lens, seq_lens
            )
            loss = crit(
                recon[pred_loss_idx], targets[target_loss_idx]
            )
            if not torch.isfinite(loss):
                print("Warning: got NaN loss, ignoring...")
                continue
            """
            # Required for loss
            target_list = []
            target_sum = 0
            for s in seq_lens:
                target_x = targets[target_sum: target_sum + s]
                target_list.append(torch_geometric.data.Data(x=target_x))
                target_sum += s
            target = torch_geometric.data.Batch.from_data_list(target_list)

            pred_list = []
            pred_sum = 0
            for i in recon_b_idx.unique():
                pred_x = recon[recon_b_idx == i]
                pred_list.append(torch_geometric.data.Data(x=pred_x))
            pred = torch_geometric.data.Batch.from_data_list(pred_list)
            if pred.num_graphs != target.num_graphs:
                breakpoint()

            loss = crit(target, pred)
            """
            pbar2.set_description(f"train loss: {loss:.3f}")
            loss.backward()
            opt.step()
            wandb.log({"loss": loss, "epoch": epoch})

        val_loss = 0
        targets = valset
        b_idx = torch.repeat_interleave(
            torch.arange(val_seq_lens.numel(), device=device), val_seq_lens
        )
        with torch.no_grad():
            recon, recon_b_idx, pred_seq_lens = model(targets, b_idx)
        """
        # Required for loss
        target_list = []
        target_sum = 0
        for s in val_seq_lens:
            target_x = targets[target_sum: target_sum + s]
            target_list.append(torch_geometric.data.Data(x=target_x))
            target_sum += s
        target = torch_geometric.data.Batch.from_data_list(target_list)

        pred_list = []
        pred_sum = 0
        for i in recon_b_idx.unique():
            pred_x = recon[recon_b_idx == i]
            pred_list.append(torch_geometric.data.Data(x=pred_x))
        pred = torch_geometric.data.Batch.from_data_list(pred_list)
        val_loss = crit(target, pred)
        """

        pred_loss_idx, target_loss_idx = get_loss_idxs(
            pred_seq_lens, val_seq_lens
        )
        val_loss = crit(
            recon[pred_loss_idx], targets[target_loss_idx]
        )

        viz = torchvision.utils.make_grid(
            torch.cat([targets[:16], recon[:16]], dim=0), nrow=16
        )
        imshow(viz, epoch)

        pbar.set_description(f"val loss: {val_loss:.3f}")
        wandb.log(
            {
                "val_loss": val_loss,
                "val_image": viz / 2 + 0.5,
            }, 
        commit=False)


if __name__ == '__main__':
    main()
