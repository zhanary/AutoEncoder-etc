import os
import time

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.utils import save_image

from model import StackedAutoEncoder

if not os.path.exists('./imgs'):
    os.mkdir('./imgs')

def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 1000
batch_size = 128

# img_transform = transforms.Compose([
#     #transforms.RandomRotation(360),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0), #改变图像亮度饱和度
#     transforms.ToTensor(),
# ])



DOWNLOAD_DATA = False      #是否下载mnist
# Mnist digits dataset
dataset = torchvision.datasets.MNIST (
    root='../mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to                                              # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_DATA,                        # download it if you don't have it
)

# dataset = MNIST('../data/mnist/', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
print("It is DAE")
model = StackedAutoEncoder().cuda()

for epoch in range(num_epochs):
    if epoch % 10 == 0:
        # Test the quality of our features with a randomly initialzed linear classifier.
        classifier = nn.Linear(10, 10).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    model.train()
    total_time = time.time()
    correct = 0
    for i, data in enumerate(dataloader):
        img, target = data
        target = Variable(target).cuda()
        img = Variable(img).cuda()

        #######change size
        img = img.view(-1, 28 * 28)  # batch x, shape (batch, 28*28)
        #print("*********", img.shape)


        features = model(img).detach()
        #print("#########", features.shape)
        prediction = classifier(features.view(features.size(0), -1))
        loss = criterion(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = prediction.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    total_time = time.time() - total_time

    model.eval()
    img, _ = data
    img = Variable(img).cuda()
    img = img.view(-1, 28 * 28)  # batch x, shape (batch, 28*28)

    features, x_reconstructed = model(img)
    reconstruction_loss = torch.mean((x_reconstructed.data - img.data)**2)

    if epoch % 10 == 0:
        print("Saving epoch {}".format(epoch))
        orig = to_img(img.cpu().data)
        save_image(orig, './imgs/orig_{}.png'.format(epoch))
        pic = to_img(x_reconstructed.cpu().data)
        save_image(pic, './imgs/reconstruction_{}.png'.format(epoch))

    print("Epoch {} complete\tTime: {:.4f}s\t\tLoss: {:.4f}".format(epoch, total_time, reconstruction_loss))
    print("Feature Statistics\tMean: {:.4f}\t\tMax: {:.4f}\t\tSparsity: {:.4f}%".format(
        torch.mean(features.data), torch.max(features.data), torch.sum(features.data == 0.0)*100 / features.data.numel())
    )
    print("Linear classifier performance: {}/{} = {:.2f}%".format(correct, len(dataloader)*batch_size, 100*float(correct) / (len(dataloader)*batch_size)))
    print("="*80)

torch.save(model.state_dict(), './CDAE.pth')
