    # -*- coding: utf-8 -*-
import os
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from torchvision.utils import save_image

plt.switch_backend('agg')  #避免plt在ssh情况下报错

if not os.path.exists('./imgs'):
    os.mkdir('./imgs')


class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim)
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return F.relu(self.linear3(x))


class VAE(torch.nn.Module):
    latent_dim = 10

    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(64, 10)
        self._enc_log_sigma = torch.nn.Linear(64, 10)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return (mu.cuda() + sigma.cuda() * Variable(std_z.cuda(), requires_grad=False))  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc.cuda())
        return self.decoder(z)


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


if __name__ == '__main__':

    input_dim = 28 * 28
    batch_size = 32

    transform = transforms.Compose(
        [transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST('../mnist/', download=False, transform=transform)

    dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
                                             shuffle=True, num_workers=8)
    print("It is VAE")
    print('Number of samples: ', len(mnist))

    encoder = Encoder(input_dim, 128, 64)
    decoder = Decoder(10, 64, 128, input_dim)
    vae = VAE(encoder, decoder).cuda()

    criterion = nn.MSELoss()

    optimizer = optim.Adam(vae.parameters(), lr=0.001)
    l = None
    for epoch in range(100):
        for i, data in enumerate(dataloader, 0):
            inputs, classes = data
            inputs, classes = Variable(inputs.resize_(batch_size, input_dim)).cuda(), Variable(classes).cuda()
            optimizer.zero_grad()
            dec = vae(inputs)
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs) + ll
            loss.backward()
            optimizer.step()
            l = loss.item()

        print(epoch, l)

        ######img
        orig = inputs.cpu().data.view(inputs.cpu().data.size(0), 1, 28, 28)
        print ("get orig")
        save_image(orig, './imgs/orig_{}.png'.format(epoch))
        pic = dec.cpu().data.view(dec.cpu().data.size(0), 1, 28, 28)
        print("get output")
        save_image(pic, './imgs/reconstruction_{}.png'.format(epoch))





        # plt.imshow(vae(inputs).data[0].numpy().reshape(28, 28), cmap='gray')
        # plt.show(block=True)
        # plt.savefig('output - %d.jpg'%epoch)
