import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler 

import numpy as np 
import syft as sy
import time
import tqdm as tqdm 

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

hook = sy.TorchHook(torch)

_ = torch.manual_seed(1234)
batch_size = 32

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

cifar10_data = datasets.MNIST('./dataset', train=True,
                              download=True, transform=transform)

cifar10_data_test = datasets.MNIST('./dataset', train=False,
                              download=True, transform=transform)

cifar10_train_loader = torch.utils.data.DataLoader(cifar10_data, batch_size=batch_size, shuffle=True)
cifar10_test_loader = torch.utils.data.DataLoader(cifar10_data_test, batch_size=batch_size, shuffle=True)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

sam = sy.VirtualWorker(hook, id="sam")
kelly = sy.VirtualWorker(hook, id="kelly")
workers = [sam, kelly]


crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

from syft.serde.compression import NO_COMPRESSION
sy.serde.compression.default_compress_scheme = NO_COMPRESSION

model = LeNet()
model.load_state_dict(torch.load('./model/MNIST.pth', map_location='cpu'))
model.eval()

encryption_kwargs = dict(
    workers=workers, 
    crypto_provider=crypto_provider, 
    protocol="fss", 
    requires_grad=False,
    precision_fractional=4,
)

first_batch, first_target = next(iter(cifar10_test_loader))

ptr_first_batch = first_batch.send(sam)
ptr_model = model.send(sam)
encrypted_model = ptr_model.encrypt(**encryption_kwargs).get()
start_time = time.time()

print('data sent')

for _ in range(1):
    encrypted_first_batch = ptr_first_batch.encrypt(**encryption_kwargs).get()
    print(encrypted_first_batch)

    print('model, data encrypted')
    print(f"Encrypting Batch Duration: {time.time() - start_time}")
    print(first_target)

    encrypted_result = encrypted_model(encrypted_first_batch)
    print(f"Inference done in {time.time() - start_time} sec. Privacy preserving fetching of prediction!")
    # print(encrypted_result.decrypt())
    pred = encrypted_result.argmax(dim=1)

    print(f"Total Duration: {time.time() - start_time}")
    # decrypt() = get().float_precision() - the specific result isn't sensible as the ResNet wasn't actually trained 
    # decrypt() = get().float_precision() - the specific result isn't sensible as the ResNet wasn't actually trained 
    # decrypt() = get().float_precision() - the specific result isn't sensible as the ResNet wasn't actually trained 
    print(f"Result: {pred.decrypt()}")