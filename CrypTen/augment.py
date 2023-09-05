import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random, os
if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from GenerateMPCAE import LeNet, MPCLeNet, MPCLeNetAUG, LeNet_internal1, LeNet_internal2, LeNet_internal3, LeNet_internal4
from PIL import Image
import glob
from examples.util import NoopContextManager
from examples.meters import AverageMeter
import crypten
import crypten.communicator as comm
import time, logging, tqdm, warnings, pickle
import crypten.mpc as mpc
from crypten.nn import model_counter
from util import augdata
import multiprocessing
from crypten.config import cfg

class Logistic(nn.Sequential):
    def __init__(self):
        super(Logistic, self).__init__()
        self.fc1 = nn.Linear(23, 120)
        self.fc2 = nn.Linear(120, 2)
        self.activation = nn.Sigmoid()
        self.internal1 = None

    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        self.internal1 = x.clone().detach()
        x = self.fc2(x)
        return x

class Logistic_internal1(nn.Sequential):
    def __init__(self):
        super(Logistic_internal1, self).__init__()
        self.fc1 = nn.Linear(23, 120)
        self.fc2 = nn.Linear(120, 2)
        self.activation = nn.Sigmoid()
        self.internal1 = None

    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        self.internal1 = x.clone().detach()
        return x

class BankNet(nn.Sequential):
    def __init__(self):
        super(BankNet, self).__init__()
        self.fc1 = nn.Linear(20, 250)
        self.fc2 = nn.Linear(250, 2)
        self.activation = nn.GELU()
        self.internal1 = None

    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        self.internal1 = x.clone().detach()
        x = self.fc2(x)
        return x

class BankNetMPC(nn.Sequential):
    def __init__(self):
        super(BankNetMPC, self).__init__()
        self.fc1 = nn.Linear(20, 250)
        self.fc2 = nn.Linear(250, 2)
        self.activation = nn.ELU()

    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class BankNetMPC_internal1(nn.Sequential):
    def __init__(self):
        super(BankNetMPC_internal1, self).__init__()
        self.fc1 = nn.Linear(20, 250)
        self.fc2 = nn.Linear(250, 2)
        self.activation = nn.ELU()

    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        return x

def stat_internal_layer1(inputs, plaintext_net, mpc_net, args):
    images = inputs[0:1]
    images = images.to(args.device)
    outputs_plaintext = plaintext_net(images)
    internal1_plaintext = plaintext_net.internal1
    stats_max = np.zeros(internal1_plaintext.flatten().shape)
    store_index_max = []
    stats_min = np.zeros(internal1_plaintext.flatten().shape)
    store_index_min = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for i in tqdm.tqdm(range(inputs.shape[0])):
        images = inputs[i:i+1]
        images = images.to(args.device)
        outputs_plaintext = plaintext_net(images)
        crypten_pred(images, mpc_net, return_dict, flag=False)

        internal1_plaintext = plaintext_net.internal1
        internal1_mpc = return_dict['output']
        err = (internal1_plaintext - internal1_mpc).abs()
        stat_num = err.flatten().shape[0] // 3
        largest, indices_max = torch.topk(err.flatten(), stat_num)
        smallest, indices_min = torch.topk(-err.flatten(), stat_num)
        indices_max = indices_max.numpy()
        indices_min = indices_min.numpy()
        for j in range(indices_max.shape[0]):
            convert_index = indices_max[j]
            stats_max[convert_index] += 1
        for j in range(indices_min.shape[0]):
            convert_index = indices_min[j]
            stats_min[convert_index] += 1
    stats_max = torch.Tensor(stats_max)
    stats_min = torch.Tensor(stats_min)
    largest, indices_max = torch.topk(stats_max.flatten(), stat_num)
    smallest, indices_min = torch.topk(stats_min.flatten(), stat_num)
    for j in range(indices_max.shape[0]):
        convert_index = indices_max[j].item()
        store_index_max.append(convert_index)
    for j in range(indices_min.shape[0]):
        convert_index = indices_min[j].item()
        store_index_min.append(convert_index)
    return store_index_max, store_index_min

def stat_internal_layer2(inputs, plaintext_net, mpc_net, args):
    images = inputs[0:1]
    images = images.to(args.device)
    outputs_plaintext = plaintext_net(images)
    internal2_plaintext = plaintext_net.internal2
    stats_max = np.zeros(internal2_plaintext.flatten().shape)
    store_index_max = []
    stats_min = np.zeros(internal2_plaintext.flatten().shape)
    store_index_min = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for i in tqdm.tqdm(range(inputs.shape[0])):
        images = inputs[i:i+1]
        images = images.to(args.device)
        outputs_plaintext = plaintext_net(images)
        crypten_pred(images, mpc_net, return_dict, flag=False)

        internal2_plaintext = plaintext_net.internal2
        internal2_mpc = return_dict['output']
        err = (internal2_plaintext - internal2_mpc).abs()
        stat_num = err.flatten().shape[0] // 3
        largest, indices_max = torch.topk(err.flatten(), stat_num)
        smallest, indices_min = torch.topk(-err.flatten(), stat_num)
        indices_max = indices_max.numpy()
        indices_min = indices_min.numpy()
        for j in range(indices_max.shape[0]):
            convert_index = indices_max[j]
            stats_max[convert_index] += 1
        for j in range(indices_min.shape[0]):
            convert_index = indices_min[j]
            stats_min[convert_index] += 1
    stats_max = torch.Tensor(stats_max)
    stats_min = torch.Tensor(stats_min)
    largest, indices_max = torch.topk(stats_max.flatten(), stat_num)
    smallest, indices_min = torch.topk(stats_min.flatten(), stat_num)
    for j in range(indices_max.shape[0]):
        convert_index = indices_max[j].item()
        store_index_max.append(convert_index)
    for j in range(indices_min.shape[0]):
        convert_index = indices_min[j].item()
        store_index_min.append(convert_index)
    return store_index_max, store_index_min

def stat_internal_layer3(inputs, plaintext_net, mpc_net, args):
    images = inputs[0:1]
    images = images.to(args.device)
    outputs_plaintext = plaintext_net(images)
    internal3_plaintext = plaintext_net.internal3
    stats_max = np.zeros(internal3_plaintext.flatten().shape)
    store_index_max = []
    stats_min = np.zeros(internal3_plaintext.flatten().shape)
    store_index_min = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for i in tqdm.tqdm(range(inputs.shape[0])):
        images = inputs[i:i+1]
        images = images.to(args.device)
        outputs_plaintext = plaintext_net(images)
        crypten_pred(images, mpc_net, return_dict, flag=False)

        internal3_plaintext = plaintext_net.internal3
        internal3_mpc = return_dict['output']
        err = (internal3_plaintext - internal3_mpc).abs()

        stat_num = err.flatten().shape[0] // 3
        largest, indices_max = torch.topk(err.flatten(), stat_num)
        smallest, indices_min = torch.topk(-err.flatten(), stat_num)
        indices_max = indices_max.numpy()
        indices_min = indices_min.numpy()
        for j in range(indices_max.shape[0]):
            convert_index = indices_max[j]
            stats_max[convert_index] += 1
        for j in range(indices_min.shape[0]):
            convert_index = indices_min[j]
            stats_min[convert_index] += 1
    stats_max = torch.Tensor(stats_max)
    stats_min = torch.Tensor(stats_min)

    largest, indices_max = torch.topk(stats_max.flatten(), stat_num)
    smallest, indices_min = torch.topk(stats_min.flatten(), stat_num)
    for j in range(indices_max.shape[0]):
        convert_index = indices_max[j].item()
        store_index_max.append(convert_index)
    for j in range(indices_min.shape[0]):
        convert_index = indices_min[j].item()
        store_index_min.append(convert_index)
    return store_index_max, store_index_min


def stat_internal_layer4(inputs, plaintext_net, mpc_net, args):
    images = inputs[0:1]
    images = images.to(args.device)
    outputs_plaintext = plaintext_net(images)
    internal4_plaintext = plaintext_net.internal4
    stats_max = np.zeros(internal4_plaintext.flatten().shape)
    store_index_max = []
    stats_min = np.zeros(internal4_plaintext.flatten().shape)
    store_index_min = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for i in tqdm.tqdm(range(inputs.shape[0])):
        images = inputs[i:i+1]
        images = images.to(args.device)
        outputs_plaintext = plaintext_net(images)
        crypten_pred(images, mpc_net, return_dict, flag=False)

        internal4_plaintext = plaintext_net.internal4
        internal4_mpc = return_dict['output']
        err = (internal4_plaintext - internal4_mpc).abs()
        stat_num = err.flatten().shape[0] // 3
        largest, indices_max = torch.topk(err.flatten(), stat_num)
        smallest, indices_min = torch.topk(-err.flatten(), stat_num)
        indices_max = indices_max.numpy()
        indices_min = indices_min.numpy()
        for j in range(indices_max.shape[0]):
            convert_index = indices_max[j]
            stats_max[convert_index] += 1
        for j in range(indices_min.shape[0]):
            convert_index = indices_min[j]
            stats_min[convert_index] += 1
    stats_max = torch.Tensor(stats_max)
    stats_min = torch.Tensor(stats_min)
    largest, indices_max = torch.topk(stats_max.flatten(), stat_num)
    smallest, indices_min = torch.topk(stats_min.flatten(), stat_num)
    for j in range(indices_max.shape[0]):
        convert_index = indices_max[j].item()
        store_index_max.append(convert_index)
    for j in range(indices_min.shape[0]):
        convert_index = indices_min[j].item()
        store_index_min.append(convert_index)
    return store_index_max, store_index_min


def compare_labels(inputs, plaintext_net, mpc_net, args):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    same_num = 0
    total_num = inputs.shape[0]
    for i in tqdm.tqdm(range(inputs.shape[0])):
        images = inputs[i:i+1]
        images = images.to(args.device)
        outputs_plaintext = plaintext_net(images)
        crypten_pred(images, mpc_net, return_dict, flag=True)
        _, predicted_plaintext = torch.max(outputs_plaintext.data, 1)
        predicted_mpc = return_dict['label']
        if predicted_mpc == predicted_plaintext:
            same_num += 1
    return same_num, total_num        

def evaluate(inputs, plaintext_model, mpc_model, mpc_aug_model, args):
    correct_mpc = 0
    correct_mpc_aug = 0
    total = 0
    ground_truth = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(inputs.shape[0])):
            images = inputs[i:i+1]
            images = images.to(args.device)
            outputs_plaintext = plaintext_model(images)
            outputs_mpc = mpc_model(images)
            outputs_mpc_aug = mpc_aug_model(images)
            _, predicted_plaintext = torch.max(outputs_plaintext.data, 1)
            _, predicted_mpc = torch.max(outputs_mpc.data, 1)
            _, predicted_mpc_aug = torch.max(outputs_mpc_aug.data, 1)
            if predicted_plaintext == predicted_mpc:
                correct_mpc += 1
            if predicted_plaintext == predicted_mpc_aug:
                correct_mpc_aug += 1
            total += 1
    return total, correct_mpc, correct_mpc_aug

@mpc.run_multiprocess(world_size=2)
def crypten_pred(inputs, model_crypten, return_dict, flag=False):
    inputCrypt = inputs.clone()
    input_size = inputCrypt.size()
    dummy_input = torch.empty(input_size)
    private_model = crypten.nn.from_pytorch(model_crypten, dummy_input).encrypt(src=0)
    private_input = crypten.cryptensor(inputCrypt, src=1)
    OutputCrypt = private_model(private_input)
    OutputCrypt = OutputCrypt.get_plain_text()
    if flag:
        _, predicted = torch.max(OutputCrypt.data, 1)
        return_dict['label'] = predicted
    return_dict['output'] = OutputCrypt

@mpc.run_multiprocess(world_size=2)
def verify_crypt(inputs, labels, args):
    modelCryp = LeNet().to(args.device)
    modelCryp.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    modelCryp.eval()

    inputCrypt = inputs[0:1]
    input_size = inputCrypt.size()
    dummy_input = torch.empty(input_size)
    correct_num = 0
    private_model = crypten.nn.from_pytorch(modelCryp, dummy_input).encrypt(src=0)
    for i in tqdm.tqdm(range(len(inputs))):
        inputCrypt = inputs[i:i+1]
        private_input = crypten.cryptensor(inputCrypt, src=1)
        OutputCrypt = private_model(private_input)
        OutputCrypt = OutputCrypt.get_plain_text()
        _, predicted = torch.max(OutputCrypt.data, 1)
        if predicted == labels[i]:
            correct_num += 1
    print(correct_num, len(inputs))


def table_to_tensor(folder_path):
    inputs = []
    for tensor_path in glob.glob(folder_path):
        cur_tensor = torch.load(tensor_path, map_location=torch.device('cpu')).numpy()
        inputs.append(cur_tensor)
    inputs = np.array(inputs)
    return inputs

def png_to_tensor(folder_path):
    inputs = []
    image_name = []
    for im_path in glob.glob(folder_path):
        image_name.append(im_path)
        im = np.asarray(Image.open(im_path).convert('L'))
        inputs.append(im)
    return np.array(inputs) / 255, image_name

def eval_testdata(net, testloader, args):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm.tqdm(testloader):
            images, labels = data
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images:',  (100 * correct / total))

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

class first_layer_weights(nn.Sequential):
    def __init__(self):
        super(first_layer_weights, self).__init__()
        self.fc = nn.Linear(1176, 2)
    
    def forward(self, x):
        return self.fc(x)

def neuron_weights(model_path, args):
    net = first_layer_weights().to(args.device)
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    stats = np.zeros((1, 6, 14, 14))
    stat_num = 1176 // 10
    with torch.no_grad():
        largest, indices1 = torch.topk(net.fc.weight[0].flatten().abs(), stat_num)
        largest, indices2 = torch.topk(net.fc.weight[1].flatten().abs(), stat_num)

        smallest, indices3 = torch.topk(-net.fc.weight[0].flatten().abs(), stat_num)
        smallest, indices4 = torch.topk(-net.fc.weight[1].flatten().abs(), stat_num)

    indices1 = indices1.numpy()
    indices2 = indices2.numpy()
    indices3 = indices3.numpy()
    indices4 = indices4.numpy()
    store_index_max = []
    store_index_min = []
    for j in range(indices1.shape[0]):
        convert_index = np.unravel_index(indices1[j], stats.shape)
        store_index_max.append(convert_index)
    for j in range(indices2.shape[0]):
        convert_index = np.unravel_index(indices2[j], stats.shape)
        store_index_max.append(convert_index)
    store_index_max = list(set(store_index_max))
    for j in range(indices3.shape[0]):
        convert_index = np.unravel_index(indices3[j], stats.shape)
        if convert_index not in store_index_max:
            store_index_min.append(convert_index)
    for j in range(indices4.shape[0]):
        convert_index = np.unravel_index(indices4[j], stats.shape)
        if convert_index not in store_index_max:
            store_index_min.append(convert_index)
    store_index_min = list(set(store_index_min))
    return store_index_max, store_index_min

def train_neuron_classifier(AugDataloader, testDataloader, OriginalNet, args):
    net = first_layer_weights().to(args.device)
    net.apply(weigth_init)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    for epoch in range(args.epoch_num):
        running_loss = 0.0
        for data in tqdm.tqdm(AugDataloader):
            images, labels = data
            images = images.to(args.device)
            labels = labels.to(args.device).type(torch.LongTensor)
            cur_batch_size = images.shape[0]
            OriginalNet(images)
            inputs = OriginalNet.internal1.view(cur_batch_size, -1).clone()
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        total = 0
        correct = 0
        for test_data in testDataloader:
            images, labels = test_data
            images = images.to(args.device)
            labels = labels.to(args.device).type(torch.LongTensor)
            cur_batch_size = images.shape[0]
            OriginalNet(images)
            inputs = OriginalNet.internal1.view(cur_batch_size, -1).clone()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('[%d] loss: %.3f ACC: %.3f' % (epoch + 1, running_loss / 5000, correct/total))
        running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), './model/AugModel_Layer_1.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epoch_num', type=int, default=25)
    parser.add_argument('--aug_layer', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dataset', type=str, default='MNIST')

    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    device = "cpu"
    args.device = device
    
    if args.dataset == 'MNIST':
        args.model_path = './model/MNIST_LeNet_sigmoid_batchnorm.pth'
        inputs, image_name = png_to_tensor('./results/LeNet12_bit/*.jpg')
        inputs = inputs.reshape(inputs.shape[0], 1, 28, 28)
        inputs = torch.from_numpy(inputs).type(torch.float)
        inputs = (inputs - 0.1307) / 0.3081

        plaintext_model = LeNet().to(args.device)
        plaintext_model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        plaintext_model.eval()

        mpc_model = LeNet().to(args.device)
        mpc_model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        mpc_model.eval()

        mpc_model_internal1 = LeNet_internal1().to(args.device)
        mpc_model_internal1.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        mpc_model_internal1.eval()

        mpc_model_internal2 = LeNet_internal2().to(args.device)
        mpc_model_internal2.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        mpc_model_internal2.eval()

        mpc_model_internal3 = LeNet_internal3().to(args.device)
        mpc_model_internal3.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        mpc_model_internal3.eval()

        mpc_model_internal4 = LeNet_internal4().to(args.device)
        mpc_model_internal4.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        mpc_model_internal4.eval()

        _path = './results/MNIST/'

    elif args.dataset == 'Credit':
        args.model_path = './model/Credit.pth'
        inputs = table_to_tensor('./results/Credit/Fuzz_AE_*.pt')
        inputs = inputs.reshape(inputs.shape[0], 23)
        inputs = torch.from_numpy(inputs).type(torch.float)

        plaintext_model = Logistic().to(args.device)
        plaintext_model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        plaintext_model.eval()

        mpc_model = Logistic().to(args.device)
        mpc_model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        mpc_model.eval()

        mpc_model_internal1 = Logistic_internal1().to(args.device)
        mpc_model_internal1.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        mpc_model_internal1.eval()

        _path = './results/Credit/'
    
    elif args.dataset == 'Bank':
        args.model_path = './model/Bank.pth'
        inputs = table_to_tensor('./results/Bank/Fuzz_AE_*.pt')
        inputs = inputs.reshape(inputs.shape[0], 20)
        inputs = torch.from_numpy(inputs).type(torch.float)

        plaintext_model = BankNet().to(args.device)
        plaintext_model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        plaintext_model.eval()

        mpc_model = BankNetMPC().to(args.device)
        mpc_model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        mpc_model.eval()

        mpc_model_internal1 = BankNetMPC_internal1().to(args.device)
        mpc_model_internal1.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        mpc_model_internal1.eval()

        _path = './results/Bank/'

    crypten.init()
    model_counter.register_counter(args)


    if args.aug_layer == 1:
        store_index_max, store_index_min = stat_internal_layer1(inputs, plaintext_model, mpc_model_internal1, args)
        with open(_path + 'layer1_max.pkl', 'wb') as fp:
            pickle.dump(store_index_max, fp)
        with open(_path + 'layer1_min.pkl', 'wb') as fp:
            pickle.dump(store_index_min, fp)
        print('Voting length: %d, %d' % (len(store_index_max), len(store_index_min)))
    elif args.aug_layer == 2:
        store_index_max, store_index_min = stat_internal_layer2(inputs, plaintext_model, mpc_model_internal2, args)
        with open(_path + 'layer2_max.pkl', 'wb') as fp:
            pickle.dump(store_index_max, fp)
        with open(_path + 'layer2_min.pkl', 'wb') as fp:
            pickle.dump(store_index_min, fp)
        print('Voting length: %d, %d' % (len(store_index_max), len(store_index_min)))
    elif args.aug_layer == 3:
        store_index_max, store_index_min = stat_internal_layer3(inputs, plaintext_model, mpc_model_internal3, args)
        with open(_path + 'layer3_max.pkl', 'wb') as fp:
            pickle.dump(store_index_max, fp)
        with open(_path + 'layer3_min.pkl', 'wb') as fp:
            pickle.dump(store_index_min, fp)
        print('Voting length: %d, %d' % (len(store_index_max), len(store_index_min)))
    elif args.aug_layer == 4:
        store_index_max, store_index_min = stat_internal_layer4(inputs, plaintext_model, mpc_model_internal4, args)
        with open(_path + 'layer4_max.pkl', 'wb') as fp:
            pickle.dump(store_index_max, fp)
        with open(_path + 'layer4_min.pkl', 'wb') as fp:
            pickle.dump(store_index_min, fp)
        print('Voting length: %d, %d' % (len(store_index_max), len(store_index_min)))
    else:
        same_nums, total_nums = compare_labels(inputs, plaintext_model, mpc_model, args)
        print(same_nums, total_nums)