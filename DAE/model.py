import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    r"""
    Convolutional denoising autoencoder layer for stacked autoencoders.
    This module is automatically trained when in model.training is True.
    Args:
        input_size: The number of features in the input
        output_size: The number of features to output
        stride: Stride of the convolutional layers.
    """

    def __init__(self, input_size, output_size):
        super(AutoEncoder, self).__init__()

        # self.forward_pass = nn.Sequential(
        #     nn.Conv2d(input_size, output_size, kernel_size=2, stride=2, padding=padding_size1),
        #     nn.ReLU(),
        # )
        self.forward_pass = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 10),
            nn.Tanh(),
        )

        # self.backward_pass = nn.Sequential(
        #     nn.ConvTranspose2d(output_size, input_size, kernel_size=2, stride=2, padding=padding_size2),
        #     nn.ReLU(),
        # )
        self.backward_pass = nn.Sequential(
            nn.Linear(10, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, output_size),
            nn.Tanh(),
        )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x):
        # Train each autoencoder individually
        x = x.detach()

        # Add noise, but use the original lossless input as the target.
        x_noisy = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        y = self.forward_pass(x_noisy)
        #y = self.forward_pass(x)

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

        self.ae = AutoEncoder(28*28, 28*28)

    def forward(self, x):
        a1 = self.ae(x)

        if self.training:
            return a1

        else:
            return a1, self.reconstruct(a1)

    def reconstruct(self, x):
        x_reconstruct = self.ae.reconstruct(x)
        return x_reconstruct