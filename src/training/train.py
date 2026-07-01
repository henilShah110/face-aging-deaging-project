import torch
from torch.utils.data import DataLoader

import os
from torchvision.utils import save_image
from tqdm import tqdm

from src.dataset.cyclegan_dataset import CycleGANDataset

from src.models.generator import Generator
from src.models.discriminator import Discriminator

from src.training.losses import (
    adversarial_loss,
    cycle_loss,
    identity_loss
)

import itertools
import torch.optim as optim

from src.training.checkpoint import (
    save_checkpoint,
    load_checkpoint
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

dataset = CycleGANDataset(
    "data/processed/young",
    "data/processed/old"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

G_YO = Generator().to(device)
G_OY = Generator().to(device)

D_old = Discriminator().to(device)
D_young = Discriminator().to(device)

optimizer_G = optim.Adam(
    itertools.chain(
        G_YO.parameters(),
        G_OY.parameters()
    ),
    lr=0.0002,
    betas=(0.5, 0.999)
)

optimizer_D_old = optim.Adam(
    D_old.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

optimizer_D_young = optim.Adam(
    D_young.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

G_YO.train()
G_OY.train()

D_old.train()
D_young.train()

os.makedirs("checkpoints",exist_ok=True)
os.makedirs("samples",exist_ok=True)

start_epoch = load_checkpoint(

    "checkpoints/latest.pth",

    G_YO,
    G_OY,

    D_old,
    D_young,

    optimizer_G,
    optimizer_D_old,
    optimizer_D_young,

    device

)
print(
    f"Starting from epoch {start_epoch}"
)

print(f"Device: {device}")

num_epochs = 100

for epoch in range(start_epoch,num_epochs):

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch+1}/{num_epochs}"
    )

    for batch_idx, (young, old) in enumerate(progress):

        young = young.to(device)
        old = old.to(device)

        fake_old = G_YO(young)
        fake_young = G_OY(old)

        recovered_young = G_OY(fake_old)
        recovered_old = G_YO(fake_young)

        identity_old = G_YO(old)
        identity_young = G_OY(young)

        pred_fake_old = D_old(fake_old)
        pred_fake_young = D_young(fake_young)

        real_target = torch.ones_like(
            pred_fake_old
        ).to(device)

        loss_adv_old = adversarial_loss(
            pred_fake_old,
            real_target
        )

        loss_adv_young = adversarial_loss(
            pred_fake_young,
            real_target
        )

        loss_adv = (
            loss_adv_old
            +
            loss_adv_young
        )

        loss_cycle = (
            cycle_loss(
                recovered_young,
                young
            )
            +
            cycle_loss(
                recovered_old,
                old
            )
        )

        loss_identity = (
            identity_loss(
                identity_old,
                old
            )
            +
            identity_loss(
                identity_young,
                young
            )
        )
        
        loss_G = (
            loss_adv
            + 10 * loss_cycle
            + 5 * loss_identity
        )

        optimizer_G.zero_grad()

        loss_G.backward()

        optimizer_G.step()

        fake_old = fake_old.detach()
        fake_young = fake_young.detach()

        fake_target = torch.zeros_like(
            pred_fake_old
        ).to(device)

        pred_real_old = D_old(old)
        pred_fake_old = D_old(fake_old)

        real_target_old = torch.ones_like(
            pred_real_old
        )

        fake_target_old = torch.zeros_like(
            pred_fake_old
        )

        loss_D_old_real = adversarial_loss(
            pred_real_old,
            real_target_old
        )

        loss_D_old_fake = adversarial_loss(
            pred_fake_old,
            fake_target_old
        )

        loss_D_old = (
            loss_D_old_real
            +
            loss_D_old_fake
        ) * 0.5

        optimizer_D_old.zero_grad()

        loss_D_old.backward()

        optimizer_D_old.step()

        pred_real_young = D_young(young)
        pred_fake_young = D_young(fake_young)

        real_target_young = torch.ones_like(
            pred_real_young
        )

        fake_target_young = torch.zeros_like(
            pred_fake_young
        )

        loss_D_young_real = adversarial_loss(
            pred_real_young,
            real_target_young
        )

        loss_D_young_fake = adversarial_loss(
            pred_fake_young,
            fake_target_young
        )

        loss_D_young = (
            loss_D_young_real
            +
            loss_D_young_fake
        ) * 0.5

        optimizer_D_young.zero_grad()

        loss_D_young.backward()

        optimizer_D_young.step()

        progress.set_postfix({
            "G": f"{loss_G.item():.3f}",
            "D_old": f"{loss_D_old.item():.3f}",
            "D_young": f"{loss_D_young.item():.3f}"
        })

    save_image(
        (young + 1) / 2,
        # we normalize this way because generator o/p is [-1,1]
        # pixel values expect [0,1]
        f"samples/epoch_{epoch}_young.png"
    )
    save_image(
        (fake_old + 1) / 2,
        f"samples/epoch_{epoch}_fake_old.png"
    )
    save_image(
        (old + 1) / 2,
        f"samples/epoch_{epoch}_old.png"
    )
    save_image(
        (fake_young + 1) / 2,
        f"samples/epoch_{epoch}_fake_young.png"
    )

    save_checkpoint(
        "checkpoints/latest.pth",
        epoch,
        G_YO,
        G_OY,
        D_old,
        D_young,
        optimizer_G,
        optimizer_D_old,
        optimizer_D_young
    )

    save_checkpoint(
        f"checkpoints/epoch_{epoch}.pth",
        epoch,
        G_YO,
        G_OY,
        D_old,
        D_young,
        optimizer_G,
        optimizer_D_old,
        optimizer_D_young
    )
    # only saving state dict so that only weights and parameters are saved
    # we dont want python to serialize the entire model


