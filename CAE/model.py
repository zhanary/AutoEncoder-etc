import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class CAutoEncoder(nn.Module):
    r"""
    Convolutional denoising autoencoder layer for stacked autoencoders.
    This module is automatically trained when in model.training is True.
    Args:
        input_size: The number of features in the input
        output_size: The number of features to output
        stride: Stride of the convolutional layers.
    """

    def __init__(self):
        super(CAutoEncoder, self).__init__()

        # self.forward_pass = nn.Sequential(
        #     nn.Conv2d(input_size, output_size, kernel_size=2, stride=2, padding=padding_size1),
        #     nn.ReLU(),
        # )
        self.forward_pass = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=2, stride=2),
            nn.ReLU(),
        )

        # self.backward_pass = nn.Sequential(
        #     nn.ConvTranspose2d(output_size, input_size, kernel_size=2, stride=2, padding=padding_size2),
        #     nn.ReLU(),
        # )
        self.backward_pass = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2, padding=2),
            nn.ReLU(),
        )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x):
        # Train each autoencoder individually
        x = x.detach()

        # Add noise, but use the original lossless input as the target.
        # x_noisy = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        # y = self.forward_pass(x_noisy)
        y = self.forward_pass(x)

        if self.training:
            x_reconstruct = self.backward_pass(y)
            #print(x_reconstruct.shape)
            #print("x.data shape,", x.data.shape)
            loss = self.criterion(x_reconstruct, Variable(x.data, requires_grad=False))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return y.detach()

    def reconstruct(self, x):
        return self.backward_pass(x)


class StackedAutoEncoder(nn.Module):
    r"""
    A stacked autoencoder made from the convolutional denoising autoencoders above.
    Each autoencoder is trained independently and at the same time.
    """

    def __init__(self):
        super(StackedAutoEncoder, self).__init__()

        self.ae = CAutoEncoder()

    def forward(self, x):
        a1 = self.ae(x)

        if self.training:
            return a1

        else:
            return a1, self.reconstruct(a1)

    def reconstruct(self, x):
        x_reconstruct = self.ae.reconstruct(x)
        return x_reconstruct