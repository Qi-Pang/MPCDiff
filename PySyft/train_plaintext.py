import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np


class LeNet(nn.Sequential):
    
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.activation = nn.Sigmoid()

    
    def forward(self, x):
        x = self.pool(self.activation(self.batchnorm1(self.conv1(x))))
        x = self.pool(self.activation(self.batchnorm2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class Logistic(nn.Sequential):
    def __init__(self):
        super(Logistic, self).__init__()
        self.fc1 = nn.Linear(23, 120)
        self.fc2 = nn.Linear(120, 2)
        self.activation = nn.Sigmoid()

    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class BankNet(nn.Sequential):
    def __init__(self):
        super(BankNet, self).__init__()
        self.fc1 = nn.Linear(20, 250)
        self.fc2 = nn.Linear(250, 2)
        self.activation = nn.GELU()

    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.1)
        m.bias.data.normal_(0, 0.1)

def train(trainloader, testloader, args, trainlabel=None, testlabel=None):
    if args.dataset == 'MNIST':
        net = LeNet().to(args.device)
        net.apply(weigth_init)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

        for epoch in range(args.epoch_num):  # loop over the dataset multiple times
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
    elif args.dataset == 'Credit':
        net = Logistic().to(args.device)
        net.apply(weigth_init)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
        for epoch in range(args.epoch_num):  # loop over the dataset multiple times
            running_loss = 0.0
            # for i, data in enumerate(trainloader, 0):
            for i in range(trainloader.shape[0]):
                inputs = trainloader[i]
                labels = trainlabel[i]
                # inputs, labels = data
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
            test_acc = evaluate(testloader, args, net, testlabel)
            net.train()
            if test_acc >= 98:
                break

        print('Finished Training')
        torch.save(net.state_dict(), args.model_path)

    elif args.dataset == 'Bank':
        net = BankNet().to(args.device)
        net.apply(weigth_init)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
        for epoch in range(args.epoch_num):  # loop over the dataset multiple times
            running_loss = 0.0
            # for i, data in enumerate(trainloader, 0):
            for i in range(trainloader.shape[0]):
                inputs = trainloader[i]
                labels = trainlabel[i]
                # inputs, labels = data
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
            test_acc = evaluate(testloader, args, net, testlabel)
            net.train()
            if test_acc >= 90:
                break

        print('Finished Training')
        torch.save(net.state_dict(), args.model_path)

def evaluate(testloader, args, net=None, testlabel=None):
    if args.dataset == 'MNIST':
        if net == None:
            net = LeNet()
            net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
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
    elif args.dataset == 'Credit':
        if net == None:
            net = Logistic()
            net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
            net = net.to(args.device)
        correct = 0
        total = 0
        net.eval()
        with torch.no_grad():
            for i in range(testloader.shape[0]):
                images = testloader[i]
                labels = testlabel[i]
                images = images.to(args.device)
                labels = labels.to(args.device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test data: %.2f %%' % (100 * correct / total))
        return 100 * correct / total
    elif args.dataset == 'Bank':
        if net == None:
            net = BankNet()
            net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
            net = net.to(args.device)
        correct = 0
        total = 0
        net.eval()
        with torch.no_grad():
            for i in range(testloader.shape[0]):
                images = testloader[i]
                labels = testlabel[i]
                images = images.to(args.device)
                labels = labels.to(args.device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test data: %.2f %%' % (100 * correct / total))
        return 100 * correct / total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    # parser.add_argument('--model_path', type=str, default='./model/MNIST.pth')
    # parser.add_argument('--model_path', type=str, default='./model/Credit.pth')
    parser.add_argument('--model_path', type=str, default='./model/Bank_tfe.pth')
    parser.add_argument('--dataset', type=str, default='Bank')

    args = parser.parse_args()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    args.device = device
    if args.dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # transform = transforms.Compose([transforms.ToTensor(), ])

        trainset = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        train(trainloader, testloader, args)
        evaluate(testloader, args)
    elif args.dataset == 'Credit':
        data_list = np.load('./dataset/Credit/credit_card_clients_data.npy')
        label_list = np.load('./dataset/Credit/credit_card_clients_label.npy')
        data_tensor = torch.from_numpy(np.array(data_list))
        label_tensor = torch.from_numpy(np.array(label_list))
        data_tensor = data_tensor.reshape((-1, args.batch_size, 23))
        label_tensor = label_tensor.reshape((-1, args.batch_size)).type(torch.LongTensor)
        train_num = int(0.8 * data_tensor.shape[0])
        train_data = data_tensor[:train_num]
        test_data = data_tensor[train_num:]
        train_label = label_tensor[:train_num]
        test_label = label_tensor[train_num:]

        train(train_data, test_data, args, train_label, test_label)
        evaluate(test_data, args, net=None, testlabel=test_label)
    elif args.dataset == 'Bank':
        data_list = np.load('./dataset/Bank/bank_data.npy')
        label_list = np.load('./dataset/Bank/bank_label.npy')
        data_tensor = torch.from_numpy(data_list)
        label_tensor = torch.from_numpy(label_list)
        data_tensor = data_tensor.reshape((-1, 20, 20))
        label_tensor = label_tensor.reshape((-1, 20)).type(torch.LongTensor)
        train_num = int(data_tensor.shape[0] - 50)
        print(train_num)
        train_data = data_tensor[:train_num]
        test_data = data_tensor[train_num:]
        train_label = label_tensor[:train_num]
        test_label = label_tensor[train_num:]

        train(train_data, test_data, args, train_label, test_label)
        evaluate(test_data, args, net=None, testlabel=test_label)