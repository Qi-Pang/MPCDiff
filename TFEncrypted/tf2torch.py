if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
import argparse, sys, json
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class GloroTorch(nn.Sequential):
    def __init__(self):
        super(GloroTorch, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2)

        self.batchnorm1 = nn.BatchNorm2d(32, eps=1e-3)
        self.batchnorm2 = nn.BatchNorm2d(64, eps=1e-3)
        self.batchnorm3 = nn.BatchNorm1d(512, eps=1e-3)
        self.batchnorm4 = nn.BatchNorm1d(512, eps=1e-3)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.sigmoid(self.batchnorm1(self.conv1(x)))
        x = F.sigmoid(self.conv2(x))
        x = F.sigmoid(self.batchnorm2(self.conv3(x)))
        x = F.sigmoid(self.conv4(x))
        torch.transpose(x, 1, 3)
        torch.transpose(x, 1, 2)
        x = x.view(-1, 1024)
        x = F.sigmoid(self.batchnorm3(self.fc1(x)))
        x = F.sigmoid(self.batchnorm4(self.fc2(x)))
        x = self.fc3(x)
        return x


class LeNet(nn.Sequential):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.batchnorm3 = nn.BatchNorm1d(120)
        self.batchnorm4 = nn.BatchNorm1d(84)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.internal1 = None
        self.internal2 = None
        self.internal3 = None
        self.internal4 = None

    def forward(self, x):
        x = F.sigmoid(self.batchnorm1(self.conv1(x)))
        self.internal1 = x.clone()
        x = self.pool(x)
        x = F.sigmoid(self.batchnorm2(self.conv2(x)))
        self.internal2 = x.clone()
        x = self.pool(x)
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous().view(-1, 5*5*16)
        x = F.sigmoid(self.batchnorm3(self.fc1(x)))
        self.internal3 = x.clone()
        x = F.sigmoid(self.batchnorm4(self.fc2(x)))
        self.internal4 = x.clone()
        x = self.fc3(x)
        return x


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.zero_()
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.zero_()
        m.bias.data.zero_()

if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--loss', type=str, default='sparse_trades_kl.1.5')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--model_path', type=str, default='../../model/MNIST_LeNet.pth')
    args = parser.parse_args()

    g = LeNet()
    plaintext_model = keras.models.load_model('../../model/MNIST_LeNet_Sigmoid.h5')
    weights = plaintext_model.get_weights()
    g.eval()

    for i in range(len(weights)):
        print(weights[i].shape)
    with torch.no_grad():
        sd = g.state_dict()
        sd['conv1.weight'] = torch.from_numpy(np.transpose(weights[0], (3, 2, 0, 1)))
        sd['conv1.bias'] = torch.from_numpy(weights[1])

        sd['batchnorm1.weight'] = torch.from_numpy(weights[2])
        sd['batchnorm1.bias'] = torch.from_numpy(weights[3])
        sd['batchnorm1.running_mean'] = torch.from_numpy(weights[4])
        sd['batchnorm1.running_var'] = torch.from_numpy(weights[5])

        sd['conv2.weight'] = torch.from_numpy(np.transpose(weights[6], (3, 2, 0, 1)))
        sd['conv2.bias'] = torch.from_numpy(weights[7])

        sd['batchnorm2.weight'] = torch.from_numpy(weights[8])
        sd['batchnorm2.bias'] = torch.from_numpy(weights[9])
        sd['batchnorm2.running_mean'] = torch.from_numpy(weights[10])
        sd['batchnorm2.running_var'] = torch.from_numpy(weights[11])

        sd['fc1.weight'] = torch.from_numpy(np.transpose(weights[12], (1, 0)))
        sd['fc1.bias'] = torch.from_numpy(weights[13])

        sd['batchnorm3.weight'] = torch.from_numpy(weights[14])
        sd['batchnorm3.bias'] = torch.from_numpy(weights[15])
        sd['batchnorm3.running_mean'] = torch.from_numpy(weights[16])
        sd['batchnorm3.running_var'] = torch.from_numpy(weights[17])

        sd['fc2.weight'] = torch.from_numpy(np.transpose(weights[18], (1, 0)))
        sd['fc2.bias'] = torch.from_numpy(weights[19])

        sd['batchnorm4.weight'] = torch.from_numpy(weights[20])
        sd['batchnorm4.bias'] = torch.from_numpy(weights[21])
        sd['batchnorm4.running_mean'] = torch.from_numpy(weights[22])
        sd['batchnorm4.running_var'] = torch.from_numpy(weights[23])

        sd['fc3.weight'] = torch.from_numpy(np.transpose(weights[24], (1, 0)))
        sd['fc3.bias'] = torch.from_numpy(weights[25])

        g.load_state_dict(sd)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        testset = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        acc = 0
        g.eval()

        for feature, label in testloader:
            outputs = g(feature)
            _, predicted = torch.max(outputs.data, 1)
            acc += (predicted == label).sum()
        print(acc)

    torch.save(g.state_dict(), args.model_path)