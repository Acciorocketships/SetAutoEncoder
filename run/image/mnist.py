import os
import torch
import tqdm
import random
from torch import nn
import torchvision
import torchvision.transforms as transforms
from sae.sae_new import AutoEncoder
from sae import get_loss_idxs, mean_squared_loss
from matplotlib import pyplot as plt
import numpy as np
import wandb

wandb_project = "sae-mnist"

def imshow(img, epoch, prefix=""):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.tight_layout()
    os.makedirs("/tmp/setae", exist_ok=True)
    plt.savefig(f"/tmp/setae/{prefix}recon_epoch_{epoch:03}.png")

def constrained_sum_sample_pos(n, total, min_num, max_num):
    """Return a randomly chosen list of n positive integers summing to total,
    where each integer is between [min_num, max_num) (not including max_num). Note
    that the final sample may sometimes be less than min_num."""
    nums = []
    running_sum = 0
    while running_sum < total:
        nums += [random.randint(min_num, max_num-1)]
        running_sum = sum(nums)
    nums[-1] = total - sum(nums[:-1])
    return nums

def constrained_sum_sample_pos2(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(random.sample(range(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]


# class CNNAE(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Input size: [batch, 3, 32, 32]
#         # Output size: [batch, 3, 32, 32]
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
#             nn.LeakyReLU(),
#             nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
#             nn.LeakyReLU(),
#             nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
#             nn.LeakyReLU(),
#             nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             nn.LeakyReLU(),
#             nn.Flatten(1),
#         )
#         self.decoder = nn.Sequential(
#             nn.Unflatten(1, (96, 2, 2)),
#             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
#             #nn.Tanh(),
#         )
#
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded


class CNNAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 7),
            nn.Flatten(1),
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 1, 1)),
            nn.ConvTranspose2d(64, 32, 7),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class SeqAE(nn.Module):
    def __init__(self, max_seq_len):
        super().__init__()
        self.cnn_ae = CNNAE()
        self.set_ae = AutoEncoder(
            dim=64, hidden_dim=max_seq_len * 64 // 2, max_n=max_seq_len, pos_mode="onehot"
        )

    def forward_old(self, x, b_idx):
        z = self.encode(x, b_idx)
        return self.decode(z)

    def forward(self, x, b_idx):
        cnn_feat = self.cnn_ae.encoder(x)
        z = self.set_ae.encoder(cnn_feat.flatten(1), b_idx)
        set_feat, batch = self.set_ae.decoder(z)
        img = self.cnn_ae.decoder(cnn_feat)
        imgs = self.cnn_ae.decoder(set_feat)
        return (imgs, batch), img


    def encode(self, x, b_idx):
        # Shape [B*t,f]
        feat = self.cnn_ae.encoder(x).flatten(1)
        z = self.set_ae.encoder(feat, b_idx)
        return z

    def decode(self, z):
        feat, batch = self.set_ae.decoder(z)
        img = self.cnn_ae.decoder(feat.reshape(-1, 96, 2, 2))
        return img, batch


def main():
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    if wandb_project is not None:
        wandb.init(entity="prorok-lab", project=wandb_project, group="perm")
    transform = transforms.Compose([transforms.ToTensor()])

    batch_size = 256
    val_batch_size = 64

    trainset = torchvision.datasets.MNIST(
        root="/tmp/datasets", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    testset = torchvision.datasets.MNIST(
        root="/tmp/datasets", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=val_batch_size, shuffle=False, num_workers=0
    )
    valset = torch.stack([testset[i][0] for i in range(64)], dim=0)

    # val_seq_lens = torch.tensor([4, 6, 8, 10, 12, 14, 10], device=device)
    val_seq_lens = torch.tensor([2, 4, 6, 8] + [4] * 11, device=device)
    seq_len_range = torch.tensor([1, 9])
    num_seqs_range = (1 / (seq_len_range / batch_size)).int().flip(0)
    model = SeqAE(seq_len_range[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = torch.nn.functional.mse_loss
    if wandb_project is not None:
        wandb.watch(model, log='all')

    # train loop
    epochs = 200
    pbar = tqdm.trange(epochs, position=0, desc="All Epochs", unit="epoch")
    loss = 0
    val_loss = 0
    iter = 1
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
            (recon, recon_b_idx), cnn_pred = model(targets, b_idx)
            seq_lens = model.set_ae.encoder.get_n()
            pred_seq_lens = model.set_ae.decoder.get_n_pred()
            perm = model.set_ae.encoder.get_x_perm()
            pred_loss_idx, target_loss_idx = get_loss_idxs(
                pred_seq_lens, seq_lens
            )
            set_loss = crit(
                recon[pred_loss_idx], targets[perm][target_loss_idx]
            )
            if torch.isnan(set_loss):
                set_loss = 0
            vars = model.set_ae.get_vars()
            size_loss = torch.mean(mean_squared_loss(vars["n_pred_logits"], vars["n"].unsqueeze(-1).detach().float()))
            cnn_loss = crit(cnn_pred, targets)
            loss = cnn_loss + set_loss + size_loss
            if not torch.isfinite(loss):
                print("Warning: got NaN loss, ignoring...")
                continue

            pbar2.set_description(f"train loss: {loss:.3f}")
            loss.backward()
            opt.step()
            if wandb_project is not None:
                wandb.log(
                    {"loss": loss, "set loss": set_loss, "size loss": size_loss, "cnn loss": cnn_loss, "epoch": epoch},
                    commit=True,
                )
            iter += 1

        val_loss = 0
        targets = valset
        b_idx = torch.repeat_interleave(
            torch.arange(val_seq_lens.numel(), device=device), val_seq_lens
        )
        with torch.no_grad():
            (recon, recon_b_idx), cnn_pred = model(targets, b_idx)

        pred_seq_lens = model.set_ae.decoder.get_n_pred()
        perm = model.set_ae.encoder.get_x_perm()
        pred_loss_idx, target_loss_idx = get_loss_idxs(
            pred_seq_lens, val_seq_lens
        )
        set_loss = crit(
            recon[pred_loss_idx], targets[perm][target_loss_idx]
        )
        cnn_loss = crit(cnn_pred, targets)
        val_loss = set_loss + cnn_loss
        perm = model.set_ae.encoder.get_x_perm()
        targets = targets[perm]
        cnn_pred = cnn_pred[perm]
        b = 20
        viz = torchvision.utils.make_grid(
            torch.cat([targets[:b], cnn_pred[:b].clamp(-1, 1), recon[:b]], dim=0), nrow=b
        )
        # imshow(viz, epoch)

        pbar.set_description(f"val loss: {val_loss:.3f}")
        if wandb_project is not None:
            wandb.log(
                {
                    "val loss": val_loss,
                    "val set loss": set_loss,
                    "val cnn loss": cnn_loss,
                    "val image": wandb.Image(viz / 2 + 0.5),
                },
                commit=False)


if __name__ == '__main__':
    main()
