import torch.cuda.amp

from model import Gen, Disc
import torch.optim as optim
import torch.nn as nn
from .dataset import get_dataloader
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
l_r_gen = 1e-3
l_r_disc = 1e-3
batch_size = 16
z_dim = 256
num_epoch = 100

dataloader = None

gen = Gen()
disc = Disc()

opt_gen = optim.Adam(gen.parameters(), lr=l_r_gen)
opt_disc = optim.Adam(disc.parameters(), lr=l_r_disc)

scalar_gen = torch.cuda.amp.GradScaler()
scalar_disc = torch.cuda.amp.GradScaler()

# start from resolution 4x4
# step = 0 => 8x8; 1 => 16x16; ... 5 => 256x256
step = -1
alpha = 1e-5
plus_every_x_epoch = 10
in_stable = True

def gradient_penalty(critic, real, fake, alpha, step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def compute_alpha(batch_idx, epoch, len_dataloader, plus_every_x_epoch, in_stable):
    if in_stable == False:
        alpha = (1 / plus_every_x_epoch * len_dataloader) * (batch_idx + epoch % plus_every_x_epoch)
    else:
        alpha = 1
    return alpha

for epoch in range(num_epoch):
    loop = tqdm(dataloader)
    for batch_idx, (x, _) in enumerate(loop):
        N = x.shape[0]
        alpha = compute_alpha(batch_idx, epoch, len(dataloader), plus_every_x_epoch, in_stable)

        with torch.cuda.amp.autocast():
            disc_real = disc(x, alpha, step)

            noise = torch.randn(N, z_dim, 1, 1)
            fake = gen(noise, alpha, step)
            disc_fake = disc(fake.detatch(), alpha, step)
            gp = gradient_penalty(disc, x, fake, alpha, step, DEVICE)

            loss_disc = - (torch.mean(disc_real) - torch.mean(disc_fake)) + gp *

        opt_disc.zero_grad()
        scalar_disc.scale(loss_disc).backward()
        scalar_disc.step(opt_disc)
        scalar_disc.update()

        with torch.cuda.amp.autocast():
            disc_fake = disc(noise, alpha, step)

            loss_gen = - torch.mean(disc_fake)

        opt_gen.zero_grad()
        scalar_gen.scale(loss_gen).backward()
        scalar_gen.step(opt_gen)
        scalar_gen.update()

        loop.set_postfix({"ep":epoch, "alpha":alpha, "step":step})

        if ((epoch - 5) % plus_every_x_epoch) and (epoch >=5) == 0:
            step += 1
            dataloader = get_dataloader(img_size=4, batch_size=batch_size,)
            if ((epoch - 5) % int(plus_every_x_epoch / 2)) and (epoch >=5) == 0:
                in_stable = not in_stable






