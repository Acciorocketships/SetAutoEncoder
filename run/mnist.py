import os
import torch
import tqdm
import random
from torch import nn
import torchvision
import torchvision.transforms as transforms
from sae.sae_inner import AutoEncoder

# from sae.sae_hyper import AutoEncoder
from sae import mse_sparse, get_loss_idxs
import torch_geometric
from matplotlib import pyplot as plt
import numpy as np
import time
import wandb


def imshow(img, epoch, prefix=""):
    img = img / 2 + 0.5  # unnormalize
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


class TFAE(nn.Module):
    def __init__(self, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.cnn_ae = CNNAE()
        self.set_ae = nn.Transformer(
            d_model=64, dim_feedforward=max_seq_len * 64, batch_first=True, nhead=1
        )

    def forward_old(self, x, b_idx):
        z = self.encode(x, b_idx)
        return self.decode(z)

    def forward(self, x, b_idx):
        cnn_feat = self.cnn_ae.encoder(x)
        uniqs, seq_lens = b_idx.unique(return_counts=True)
        mask = torch.ones(
            uniqs.numel(), self.max_seq_len, self.max_seq_len, dtype=torch.bool
        )

        src_in = torch.zeros(uniqs.numel(), self.max_seq_len, 64)
        cnn_feat = cnn_feat.flatten(1)
        pad_mask = torch.zeros(uniqs.numel(), self.max_seq_len)
        for u, c in zip(uniqs, seq_lens):
            mask[u, :c, :c] = False
            pad_mask[u, :c] = False
            src_in[u, :c] = cnn_feat[b_idx == u]
        tgt_in = torch.zeros_like(src_in)

        # TODO: This gets to see all tokens
        z = self.set_ae(
            src_in,
            tgt_in,  # src_mask=mask, tgt_mask=mask, memory_mask=mask,
            src_key_padding_mask=pad_mask,
            tgt_key_padding_mask=pad_mask,
        )
        batch = b_idx
        pred_seq_lens = seq_lens
        img = self.cnn_ae.decoder(cnn_feat)
        imgs = self.cnn_ae.decoder(z.reshape(-1, 64)[b_idx])
        return (imgs, batch, pred_seq_lens), img

    def encode(self, x, b_idx):
        # Shape [B*t,f]
        feat = self.cnn_ae.encoder(x).flatten(1)
        z = self.set_ae.encoder(feat, b_idx)
        return z

    def decode(self, z):
        feat, batch, pred_seq_lens = self.set_ae.decoder(z)
        img = self.cnn_ae.decoder(feat.reshape(-1, 96, 2, 2))
        return img, batch, pred_seq_lens


class SeqAE(nn.Module):
    def __init__(self, max_seq_len):
        super().__init__()
        self.cnn_ae = CNNAE()
        self.set_ae = AutoEncoder(
            dim=64,
            hidden_dim=max_seq_len * 64 // 2,
            max_n=max_seq_len,
            pe="onehot",
            data_batch=False,
        )

    def loss(self, cnn_x, set_z_pred, cnn_x_pred=None, cnn_weight=1.0):
        """Computes combined loss given x and reconstructed x.
        Note that although cnn_x and var[x] should be the same,
        var[x] is in fact a permuted version. So do not mess up
        var[x] and x."""
        var = self.set_ae.get_vars()
        valid_batch_mask = (var["n_pred_hard"] == var["n"]).reshape(-1)
        pct_correct_sizes = valid_batch_mask.float().sum() / valid_batch_mask.numel()
        pred_idx, tgt_idx = get_loss_idxs(var["n_pred_hard"], var["n"])

        set_mse_loss = torch.nn.functional.mse_loss(
            self.cnn_ae.decoder(set_z_pred)[pred_idx], cnn_x[var["x_perm_idx"]][tgt_idx]
        )
        # Sometimes mse loss can be zero if we predict entirely wrong batches
        # or predict all batches of size zero
        if torch.isnan(set_mse_loss):
            set_mse_loss = 0
        set_ce_loss = torch.nn.functional.cross_entropy(var["n_pred"], var["n"])
        loss = set_mse_loss + set_ce_loss

        if cnn_x_pred is None:
            return loss, (set_mse_loss, set_ce_loss, 0, pct_correct_sizes)

        cnn_loss = torch.nn.functional.mse_loss(cnn_x, cnn_x_pred)
        loss = loss + cnn_loss * cnn_weight
        return loss, (set_mse_loss, set_ce_loss, cnn_loss, pct_correct_sizes)

    def forward(self, x, b_idx):
        cnn_feat = self.cnn_ae.encoder(x)
        set_z_pred, set_batch_pred = self.set_ae(cnn_feat, b_idx)
        #set_x_pred = self.cnn_ae.decoder(set_z_pred)
        cnn_x_pred = self.cnn_ae.decoder(cnn_feat)
        return set_z_pred, cnn_x_pred


def main():
    device = "cpu"
    wandb.init(project="sae_cifar10")
    transform = transforms.Compose([transforms.ToTensor()])

    batch_size = 512
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
    val_seq_lens = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15], device=device)
    
    seq_len_range = torch.tensor([1, 16], device=device)
    num_seqs_range = (1 / (seq_len_range / batch_size)).int().flip(0)
    model = SeqAE(seq_len_range[1]).to(device)
    print(model)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001)
    graph = wandb.watch(model, log="gradients", log_freq=500)

    # train loop
    epochs = 200
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
            num_seqs = random.randint(*num_seqs_range)
            seq_lens = torch.tensor(
                constrained_sum_sample_pos(num_seqs, targets.shape[0], *seq_len_range),
                dtype=torch.int64,
                device=device,
            )
            assert torch.all(seq_lens > 0)
            b_idx = torch.repeat_interleave(
                torch.arange(seq_lens.numel(), device=device), seq_lens
            )
            set_pred, cnn_pred = model(targets, b_idx)
            cnn_weight = 1.0 #1 - epoch / epochs
            loss, (set_mse_loss, set_ce_loss, cnn_loss, pct_correct_sizes) = model.loss(
                targets, set_pred, cnn_pred, cnn_weight
            )
            if not torch.isfinite(loss):
                print("Warning: got NaN loss, ignoring...")
                continue

            pbar2.set_description(f"train loss: {loss:.3f}")
            loss.backward()
            opt.step()
            opt.zero_grad()
            wandb.log(
                {
                    "loss": loss,
                    "reconstruction loss": set_mse_loss,
                    "set size loss": set_ce_loss,
                    "cnn reconstruction loss": cnn_loss,
                    "percent correct set size preds": pct_correct_sizes,
                    "epoch": epoch,
                }
            )
        val_loss = 0
        targets = valset
        b_idx = torch.repeat_interleave(
            torch.arange(val_seq_lens.numel(), device=device), val_seq_lens
        )
        with torch.no_grad():
            set_pred, cnn_pred = model(targets, b_idx)
            val_loss, (set_mse_loss, set_ce_loss, cnn_loss, pct_correct_sizes) = model.loss(
                targets, set_pred, cnn_pred, cnn_weight
            )
        var = model.set_ae.get_vars()

        viz = torchvision.utils.make_grid(
            torch.cat([
                targets[var["x_perm_idx"]], 
                cnn_pred[var["x_perm_idx"]].clamp(-1, 1), 
                model.cnn_ae.decoder(set_pred[:64]).clamp(-1, 1)
            ], dim=0),
            nrow=64,
        )

        pbar.set_description(f"val loss: {val_loss:.3f}")
        wandb.log(
            {
                "val loss": val_loss,
                "val reconstruction loss": set_mse_loss,
                "val set size loss": set_ce_loss,
                "val cnn reconstruction loss": cnn_loss,
                "val image": wandb.Image((viz + 1) / 2),
                "val percent correct set size preds": pct_correct_sizes,
            },
            commit=False,
        )


if __name__ == "__main__":
    main()
