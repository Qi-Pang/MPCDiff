import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
import argparse

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

    
    def forward(self, x):
        x = self.pool(F.sigmoid(self.batchnorm1(self.conv1(x))))
        x = self.pool(F.sigmoid(self.batchnorm2(self.conv2(x))))
        x = x.view(-1, 5*5*16)
        x = F.sigmoid(self.batchnorm3(self.fc1(x)))
        x = F.sigmoid(self.batchnorm4(self.fc2(x)))
        x = self.fc3(x)
        return x

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
        x = x.view(-1, 1024)
        x = F.sigmoid(self.batchnorm3(self.fc1(x)))
        x = F.sigmoid(self.batchnorm4(self.fc2(x)))
        x = self.fc3(x)
        return x

def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

def train(trainloader, testloader, args):
    net = GloroTorch().to(args.device)
    net.apply(weigth_init)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(args.epoch_num):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        net.eval()
        test_acc = evaluate(testloader, args, net)
        net.train()
        if test_acc >= 98:
            break

    print('Finished Training')
    torch.save(net.state_dict(), args.model_path)

def evaluate(testloader, args, net=None):
    if net == None:
        net = GloroTorch()
        net.load_state_dict(torch.load(args.model_path))
        net = net.to(args.device)
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))
    return 100 * correct / total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_path', type=str, default='./model/MNIST_gloro_standard.pth')
    args = parser.parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    args.device = device

    transform = transforms.Compose([transforms.ToTensor(), ])

    trainset = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    train(trainloader, testloader, args)
    evaluate(testloader, args)
