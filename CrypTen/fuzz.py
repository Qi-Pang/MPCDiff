if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import argparse
from GenerateMPCAE import LeNet
from train_plaintext import GloroTorch
from PIL import Image
import glob
import numpy as np
from examples.util import NoopContextManager
from examples.meters import AverageMeter
import crypten
import crypten.communicator as comm
import time, logging, tqdm, warnings
import crypten.mpc as mpc
from crypten.nn import model_counter
import multiprocessing, copy
from torchvision.utils import save_image
torch.set_num_threads(1)

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

class fuzzer():
    def __init__(self, args):
        self.pool = []
        self.errs = []
        self.AEs = []
        self.true_labels = []
        self.mutation_times = []
        self.original_inputs = []
        self.randomseed = 0
        if args.dataset == 'MNIST':
            self.mutation_scale = 0.05
            self.mutation_th = 10
        elif args.dataset == 'Credit' or args.dataset == 'Bank':
            self.mutation_scale = 0.1
            self.mutation_th = 30
        self.args = args
        self.criterion = nn.CrossEntropyLoss()

    def test_mpc(self, model, testloader, args):
        count = 0
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        pbar = tqdm.tqdm(total=args.pool_size)
        crypten_acc = 0
        for data in testloader:
            count += 1
            images, labels = data
            images = images.to(args.device)
            crypten_pred(images, model, return_dict)

            if return_dict['label'] == labels:
                crypten_acc += 1
            pbar.update(1)
            pbar.set_postfix({'ACC': crypten_acc / count})
            if count >= args.pool_size:
                break
        print('CrypTen ACC:', crypten_acc / count)
        return crypten_acc / count

    def initialize(self, model, testloader, args, plain_net=None):
        count = 0
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        pbar = tqdm.tqdm(total=args.pool_size)
        plaintext_acc = 0
        crypten_acc = 0
        for data in testloader:
            count += 1
            images, labels = data
            images = images.to(args.device)
            if self.args.dataset == 'Bank':
                a, b = plaintext_pred(images, plain_net)
            else:
                a, b = plaintext_pred(images, model)
            crypten_pred(images, model, return_dict)
            if b == labels:
                plaintext_acc += 1
            if return_dict['label'] == labels:
                crypten_acc += 1
            if return_dict['label'] != b and b == labels:
                self.AEs.append(images.clone())
                if not args.test_only:
                    saving(images, len(self.AEs), self.args, original=False)
            else:
                current_err = torch.norm(a - return_dict['output'])
                self.pool.append(images.clone())
                self.errs.append(copy.deepcopy(current_err))
                self.true_labels.append(labels.clone())
                self.mutation_times.append(1)
            pbar.update(1)
            pbar.set_postfix({'AEs': len(self.AEs), 'Pool': len(self.pool)})
            if count >= args.pool_size:
                break
        print('Plaintext ACC:', plaintext_acc / count, 'CrypTen ACC:', crypten_acc / count)
        self.original_inputs = copy.deepcopy(self.pool)

    def mutation(self, seed_X, index):
        self.mutation_times[index] += 1
        return self.projection(seed_X + torch.normal(0, self.mutation_scale, size=seed_X.size()))

    def projection(self, inputs, predefined_min=-0.4242, predefined_max=2.8214):
        if self.args.dataset == 'Credit' or self.args.dataset == 'Bank':
            return torch.clamp(inputs, min=0.0, max=1.0)
        inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min()) * (predefined_max - predefined_min) + predefined_min
        return inputs

    def getseed(self):
        if self.args.guide == 'err':
            index = np.random.choice(list(range(len(self.pool))), 1)[0]
        elif self.args.guide == 'random':
            index = np.random.choice(list(range(len(self.pool))), 1)[0]
        return self.pool[index], self.errs[index], self.true_labels[index], index

    def update(self, update_X, update_err, index):
        if self.args.guide == 'err':
            if self.errs[index] < update_err:
                self.pool[index] = update_X
                self.errs[index] = update_err
                return True
            elif self.mutation_times[index] > self.mutation_th:
                self.pop(index)
        elif self.args.guide == 'random':
            self.pool[index] = update_X
            if self.mutation_times[index] > self.mutation_th:
                self.pop(index)
            else:
                return True
        return False

    def pop(self, index):
        del self.pool[index]
        del self.errs[index]
        del self.mutation_times[index]
        del self.true_labels[index]
        del self.original_inputs[index]

    def start(self, model, testloader, args, plain_net=None):
        _path = './results/' + self.args.dataset + '/'
        self.initialize(model, testloader, args, plain_net=plain_net)
        if args.repaired:
            file_name = _path + 'repaired_fuzz_results.txt'
        else:
            file_name = _path + 'fuzz_results.txt'

        fuzz_start_time = time.time()
        fuzz_end_time = time.time()
        pbar = tqdm.tqdm(total=15000)
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        mutation_num = 0
        while mutation_num < 15000 and len(self.pool) > 0:
            mutation_num += 1
            images, current_err, true_label, index = self.getseed()
            new_images = self.mutation(images, index)
            if self.args.dataset == 'Bank':
                a, b = plaintext_pred(new_images, plain_net)
            else:
                a, b = plaintext_pred(new_images, model)
            crypten_pred(new_images, model, return_dict)
            if b != return_dict['label'] and b == true_label:
                self.AEs.append(new_images)
                saving(new_images, len(self.AEs), self.args, original=False)
                saving(self.original_inputs[index], len(self.AEs), self.args, original=True)
                self.pop(index)
                with open(file_name, 'a') as txt_file:
                    txt_file.write('%d, \t%f, %d\n'%(len(self.AEs), time.time() - fuzz_start_time, mutation_num))
            else:
                new_err = torch.norm(a - return_dict['output'])
                self.update(new_images, new_err, index)
            pbar.update((time.time() - fuzz_end_time))
            fuzz_end_time = time.time()
            pbar.set_postfix({'AEs': len(self.AEs), 'Pool': len(self.pool), 'Mutation': mutation_num})


def plaintext_pred(inputs, model_plaintext):
    with torch.no_grad():
        outputs = model_plaintext(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return outputs, predicted

@mpc.run_multiprocess(world_size=2)
def crypten_pred(inputs, model_crypten, return_dict):
    inputCrypt = inputs.clone()
    input_size = inputCrypt.size()
    dummy_input = torch.empty(input_size)
    private_model = crypten.nn.from_pytorch(model_crypten, dummy_input).encrypt(src=0)

    private_input = crypten.cryptensor(inputCrypt, src=1)
    OutputCrypt = private_model(private_input)
    OutputCrypt = OutputCrypt.get_plain_text()
    _, predicted = torch.max(OutputCrypt.data, 1)
    return_dict['output'] = OutputCrypt
    return_dict['label'] = predicted

def saving(image, index, args, original=False):
    _path = './results/' + args.dataset + '/'
    if args.repaired:
        _path = _path + 'repaired_'

    if original:
        img_name = _path + 'Fuzz_Ori_' + str(index)
    else:
        img_name = _path + 'Fuzz_AE_' + str(index)

    if args.dataset == 'Credit' or args.dataset == 'Bank':
        img_name = img_name + '.pt'
        torch.save(image, img_name)
    else:
        img_name = img_name + '.jpg'
        image = image * 0.3081 + 0.1307
        save_image(image, img_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pool_size', type=int, default=1000)
    parser.add_argument('--guide', type=str, default='err')
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--repaired', action='store_true')
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    device = "cpu"
    args.device = device

    if args.dataset == 'MNIST':
        args.model_path = './model/MNIST_LeNet_sigmoid_batchnorm.pth'
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        testset = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        crypten.init()
        model_counter.register_counter(args)
        from crypten.config import cfg
        net = LeNet()
        net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        net.eval()
        if args.test_only:
            file_name = './results/MNIST/precision_bit.txt'
            precision_bits = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
            for bit in precision_bits:
                print('current bit', bit)
                cfg.encoder.precision_bits = bit
                mpc_fuzzer = fuzzer(args)
                acc = mpc_fuzzer.test_mpc(net, testloader, args)
                with open(file_name, 'a') as txt_file:
                    txt_file.write('%d, \t%f\n'%(bit, acc))
        else:
            mpc_fuzzer = fuzzer(args)
            mpc_fuzzer.start(net, testloader, args)

    elif args.dataset == 'Credit':
        args.model_path = './model/Credit.pth'
        data_list = np.load('./dataset/Credit/credit_card_clients_data.npy')
        label_list = np.load('./dataset/Credit/credit_card_clients_label.npy')
        data_tensor = torch.from_numpy(np.array(data_list))
        label_tensor = torch.from_numpy(np.array(label_list)).type(torch.LongTensor)
        test_data = data_tensor[-args.pool_size:].reshape((-1, 1, 23))
        test_label = label_tensor[-args.pool_size:].reshape((-1, 1))
        testloader = []
        for i in range(test_data.shape[0]):
            testloader.append((test_data[i], test_label[i]))

        crypten.init()
        model_counter.register_counter(args)
        from crypten.config import cfg
        net = Logistic()
        net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        net.eval()

        if args.test_only:
            file_name = './results/Credit/precision_bit.txt'
            precision_bits = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
            for bit in precision_bits:
                print('current bit', bit)
                cfg.encoder.precision_bits = bit
                mpc_fuzzer = fuzzer(args)
                acc = mpc_fuzzer.test_mpc(net, testloader, args)
                with open(file_name, 'a') as txt_file:
                    txt_file.write('%d, \t%f\n'%(bit, acc))
        else:
            mpc_fuzzer = fuzzer(args)
            mpc_fuzzer.start(net, testloader, args)

    elif args.dataset == 'Bank':
        args.model_path = './model/Bank.pth'
        data_list = np.load('./dataset/Bank/bank_data.npy')
        label_list = np.load('./dataset/Bank/bank_label.npy')
        data_tensor = torch.from_numpy(data_list)
        label_tensor = torch.from_numpy(label_list).type(torch.LongTensor)
        test_data = data_tensor[-args.pool_size:].reshape((-1, 1, 20))
        test_label = label_tensor[-args.pool_size:].reshape((-1, 1))
        testloader = []
        for i in range(test_data.shape[0]):
            testloader.append((test_data[i], test_label[i]))

        crypten.init()
        model_counter.register_counter(args)
        from crypten.config import cfg
        mpc_net = BankNetMPC()
        mpc_net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        mpc_net.eval()

        plain_net = BankNet()
        plain_net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        plain_net.eval()

        if args.test_only:
            file_name = './results/Bank/precision_bit.txt'
            precision_bits = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
            for bit in precision_bits:
                print('current bit', bit)
                cfg.encoder.precision_bits = bit
                mpc_fuzzer = fuzzer(args)
                acc = mpc_fuzzer.test_mpc(mpc_net, testloader, args)
                with open(file_name, 'a') as txt_file:
                    txt_file.write('%d, \t%f\n'%(bit, acc))
        else:
            mpc_fuzzer = fuzzer(args)
            mpc_fuzzer.start(mpc_net, testloader, args, plain_net)