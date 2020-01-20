import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def weights_init(m):
    # custom weights initialization called on netG and netD
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def test(netG, data, device):
    # Calculate average PSNR for whole dataset
    avg_psnr = 0
    with torch.no_grad():
        for batch in data:
            fake = netG(batch['low_resolution'].to(device)).detach()
            real = batch['high_resolution'].to(device)

            mse = torch.mean((fake - real) ** 2)
            psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))  ## fake and real have range [0, 255]
            avg_psnr += psnr
    avr_psnr = avg_psnr / len(data)
    print('Average PSNR: %2.2f' % avr_psnr)
    return avr_psnr

def checkpoint(netG, netD, path2save):
    # Save models
    if not os.path.exists(path2save):
        os.makedirs(path2save)
    torch.save(netG, os.path.join(path2save, "modelG.pth"))
    torch.save(netD, os.path.join(path2save, "modelD.pth"))
    print('Checkpoint saved to %s' % path2save)

def create_imgs_grid(img_batch):
    # Create images grid 4 x 4 from batch
    return np.transpose(torchvision.utils.make_grid(img_batch[:16], nrow=4, padding=2, normalize=True).cpu(), (1,2,0))

def plot_training_images(batch, device):
    # Plot high vs low resolution images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("High resolution images")
    plt.imshow(create_imgs_grid(batch['high_resolution'].to(device)))

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Low resolution images")
    plt.imshow(create_imgs_grid(batch['low_resolution'].to(device)))
    plt.show()

def plot_loss(G_losses, D_losses):
    """
    Plot loss during training
    G_losses : list of float
    D_losses : list of float
    """
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_animation(img_list):
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(i, animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    plt.show()

def plot_real_vs_fake_images(batch, fake, device):
    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(create_imgs_grid(batch['high_resolution'].to(device)))

    # Plot the fake images
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(create_imgs_grid(fake))
    plt.show()
