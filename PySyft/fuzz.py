if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import argparse
import numpy as np
import time, tqdm, warnings, copy, pickle
from torchvision.utils import save_image

import syft as sy

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

class BankNet_torch(nn.Sequential):
    def __init__(self):
        super(BankNet_torch, self).__init__()
        self.fc1 = nn.Linear(20, 250)
        self.fc2 = nn.Linear(250, 2)
        self.activation = nn.GELU()

    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class BankNet_syft(nn.Sequential):
    def __init__(self):
        super(BankNet_syft, self).__init__()
        self.fc1 = nn.Linear(20, 250)
        self.fc2 = nn.Linear(250, 2)
        self.activation = nn.Sigmoid()

    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class fuzzer():
    def __init__(self, model_p, model_enc, enc_params, workers, args):
        self.pool = []
        self.errs = []
        self.AEs = []
        self.true_labels = []
        self.mutation_times = []
        self.randomseed = 0
        if args.dataset == 'MNIST':
            self.mutation_scale = 0.05
            self.mutation_th = 10
        elif args.dataset == 'Credit':
            self.mutation_scale = 0.1
            self.mutation_th = 30
        elif args.dataset == 'Bank':
            self.mutation_scale = 0.1
            self.mutation_th = 30
        self.model_p = model_p
        self.model_enc = model_enc
        self.enc_params = enc_params
        self.workers = workers
        self.args = args
        self.choose_pool = []
        self.original_inputs = []
        self.mutation_num = 0

    def _continue(self):
        _path = './results/' + self.args.dataset + '/continue/'
        if self.args.repaired:
            _path = _path + 'repaired_'
        with open(_path + 'pool.pkl', "rb") as f:
            self.pool = pickle.load(f)
        with open(_path + 'errs.pkl', "rb") as f:
            self.errs = pickle.load(f)
        with open(_path + 'AEs.pkl', "rb") as f:
            self.AEs = pickle.load(f)
        with open(_path + 'true_labels.pkl', "rb") as f:
            self.true_labels = pickle.load(f)
        with open(_path + 'mutation_times.pkl', "rb") as f:
            self.mutation_times = pickle.load(f)
        with open(_path + 'choose_pool.pkl', "rb") as f:
            self.choose_pool = pickle.load(f)
        self.original_inputs = copy.deepcopy(self.pool)
        print('loaded!', len(self.choose_pool), 'seeds.')
        self.mutation_num = self.args.times * self.args.total_mutation

    def load_saved(self):
        _path = './results/' + self.args.dataset + '/'
        with open(_path + 'pool.pkl', "rb") as f:
            self.pool = pickle.load(f)
        with open(_path + 'errs.pkl', "rb") as f:
            self.errs = pickle.load(f)
        with open(_path + 'AEs.pkl', "rb") as f:
            self.AEs = pickle.load(f)
        with open(_path + 'true_labels.pkl', "rb") as f:
            self.true_labels = pickle.load(f)
        with open(_path + 'mutation_times.pkl', "rb") as f:
            self.mutation_times = pickle.load(f)
        self.choose_pool = list(range(len(self.pool)))
        self.original_inputs = copy.deepcopy(self.pool)
        print('loaded!', len(self.pool), 'seeds.')

    def initialize(self, testloader):
        _path = './results/' + self.args.dataset + '/'
        if self.args.repaired:
            _path = _path + 'repaired_'
        count = 0
        pbar = tqdm.tqdm(total=self.args.pool_size)
        acc_plaintext = 0
        acc_mpc = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images_to_send = images.clone().detach()
                count += images.shape[0]
                a, b = plaintext_pred(images, self.model_p)
                acc_plaintext += (b == labels).sum().item()
                ptr_batch = images_to_send.send(self.workers[0])
                encrypted_batch = ptr_batch.encrypt(**self.enc_params).get()
                encrypted_result = self.model_enc(encrypted_batch).decrypt()
                _, encrypted_pred = torch.max(encrypted_result.data, 1)
                acc_mpc += (encrypted_pred == labels).sum().item()
                for i in range(encrypted_result.shape[0]):
                    if encrypted_pred[i] != b[i] and b[i] == labels[i]:
                        self.AEs.append(images[i].clone())
                        saving(images[i], len(self.AEs), self.args, original=False)
                    else:
                        current_err = torch.norm(a[i] - encrypted_result[i])
                        self.pool.append(images[i:i+1].clone())
                        self.errs.append(copy.deepcopy(current_err))
                        self.true_labels.append(labels[i].clone())
                        self.mutation_times.append(1)
                pbar.update(images.shape[0])
                pbar.set_postfix({'AEs': len(self.AEs), 'Pool': len(self.pool)})
                if count >= self.args.pool_size:
                    break
        acc_plaintext /= count
        acc_mpc /= count
        print('Plaintext ACC', acc_plaintext, 'MPC ACC', acc_mpc)
        with open(_path + 'pool.pkl', "wb") as f:
            pickle.dump(self.pool, f)
        with open(_path + 'errs.pkl', "wb") as f:
            pickle.dump(self.errs, f)
        with open(_path + 'AEs.pkl', "wb") as f:
            pickle.dump(self.AEs, f)
        with open(_path + 'true_labels.pkl', "wb") as f:
            pickle.dump(self.true_labels, f)
        with open(_path + 'mutation_times.pkl', "wb") as f:
            pickle.dump(self.mutation_times, f)
        print('Saved..')
        self.choose_pool = list(range(len(self.pool)))
        self.original_inputs = copy.deepcopy(self.pool)

    def test_mpc(self, testloader):
        count = 0
        pbar = tqdm.tqdm(total=self.args.pool_size)
        acc_mpc = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images_to_send = images.clone().detach()
                count += images.shape[0]
                ptr_batch = images_to_send.send(self.workers[0])
                encrypted_batch = ptr_batch.encrypt(**self.enc_params).get()
                encrypted_result = self.model_enc(encrypted_batch).decrypt()
                _, encrypted_pred = torch.max(encrypted_result.data, 1)
                acc_mpc += (encrypted_pred == labels).sum().item()

                pbar.update(images.shape[0])
                pbar.set_postfix({'ACC': acc_mpc / count})
                if count >= self.args.pool_size:
                    break
        acc_mpc /= count
        print('MPC ACC', acc_mpc)
        return acc_mpc

    def mutation(self, seed_X, index):
        for i in index:
            self.mutation_times[i] += 1
        return self.projection(seed_X + torch.normal(0, self.mutation_scale, size=seed_X.size()))

    def projection(self, inputs, predefined_min=-0.4242, predefined_max=2.8214):
        if self.args.dataset == 'Credit' or self.args.dataset == 'Bank':
            predefined_min = 0.0
            predefined_max = 1.0
            return torch.clamp(inputs, min=predefined_min, max=predefined_max)
        for i in range(inputs.shape[0]):
            inputs[i] = (inputs[i] - inputs[i].min()) / (inputs[i].max() - inputs[i].min()) * (predefined_max - predefined_min) + predefined_min
        return inputs

    def getseed(self):
        get_seeds = []
        errs = []
        true_labels = []
        index = np.random.choice(self.choose_pool, min(self.args.batch_size, len(self.choose_pool)), replace=False)
        for i in range(self.args.batch_size):
            get_seeds.append(self.pool[index[i]])
            errs.append(self.errs[index[i]])
            true_labels.append(self.true_labels[index[i]])
        return get_seeds, errs, true_labels, index

    def update(self, update_X, update_err, index):
        if self.errs[index] < update_err:
            self.errs[index] = update_err
            self.pool[index] = update_X
            return True
        elif self.mutation_times[index] > self.mutation_th:
            self.pop(index)
        return False

    def pop(self, index):
        self.choose_pool.remove(index)

    def start(self, testloader):
        _path = './results/' + self.args.dataset + '/'
        if self.args.repaired:
            _path = _path + 'repaired_'
        if self.args.times == 0:
            self.initialize(testloader)
        else:
            self._continue()
        fuzz_start_time = time.time()
        fuzz_end_time = time.time()
        pbar = tqdm.tqdm(total=self.args.total_mutation)
        file_name = _path + 'fuzz_results.txt'
        mutation_num = 0
        while mutation_num < self.args.total_mutation and len(self.choose_pool) > 0:
            images, current_err, true_label, index = self.getseed()
            images = torch.cat(images, dim=0)
            new_images = self.mutation(images, index)
            images_to_send = new_images.clone().detach()
            a, b = plaintext_pred(new_images, self.model_p)
            mutation_num += images.shape[0]
            ptr_batch = images_to_send.send(self.workers[0])
            encrypted_batch = ptr_batch.encrypt(**self.enc_params).get()
            encrypted_result = self.model_enc(encrypted_batch).decrypt()
            _, encrypted_pred = torch.max(encrypted_result.data, 1)

            for i in range(encrypted_pred.shape[0]):
                if encrypted_pred[i] != b[i] and b[i] == true_label[i]:
                    self.AEs.append(new_images[i].clone())
                    self.pop(index[i])
                    saving(new_images[i].clone(), len(self.AEs), self.args, original=False)
                    saving(self.original_inputs[index[i]], len(self.AEs), self.args, original=True)
                    with open(file_name, 'a') as txt_file:
                        txt_file.write('%d, \t%f, %d\n'%(len(self.AEs), time.time() - fuzz_start_time, self.mutation_num + mutation_num))
                else:
                    new_err = torch.norm(a[i] - encrypted_result[i])
                    self.update(new_images[i:i+1], new_err, index[i])

            pbar.update(images.shape[0])
            fuzz_end_time = time.time()
            pbar.set_postfix({'AEs': len(self.AEs), 'Pool': len(self.choose_pool), 'Mutation': self.mutation_num + mutation_num})
        if self.args.repaired:
            _path_cont = './results/' + self.args.dataset + '/continue/repaired_'
        else:
            _path_cont = './results/' + self.args.dataset + '/continue/'
        with open(_path_cont + 'pool.pkl', "wb") as f:
            pickle.dump(self.pool, f)
        with open(_path_cont + 'errs.pkl', "wb") as f:
            pickle.dump(self.errs, f)
        with open(_path_cont + 'AEs.pkl', "wb") as f:
            pickle.dump(self.AEs, f)
        with open(_path_cont + 'true_labels.pkl', "wb") as f:
            pickle.dump(self.true_labels, f)
        with open(_path_cont + 'mutation_times.pkl', "wb") as f:
            pickle.dump(self.mutation_times, f)
        with open(_path_cont + 'choose_pool.pkl', "wb") as f:
            pickle.dump(self.choose_pool, f)
        print('Saved..')

def plaintext_pred(inputs, model_plaintext):
    with torch.no_grad():
        outputs = model_plaintext(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return outputs, predicted

def saving(image, index, args, original=False):
    _path = './results/' + args.dataset + '/'
    if args.repaired:
        _path = _path + 'repaired_'
    if original:
        img_name = _path + 'Fuzz_Ori_' + str(index)
    else:
        img_name = _path + 'Fuzz_AE_' + str(index)

    if args.dataset == 'MNIST':
        img_name = img_name + '.jpg'
        image = image * 0.3081 + 0.1307
        # print(image.max(), image.min())
        save_image(image, img_name)
    elif args.dataset == 'Credit' or args.dataset == 'Bank':
        img_name = img_name + '.pt'
        torch.save(image, img_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--pool_size', type=int, default=2000)
    parser.add_argument('--times', type=int, default=0)
    parser.add_argument('--total_mutation', type=int, default=2000)
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--repaired', action='store_true')
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    device = "cpu"
    args.device = device

    sy.register_counter(args)

    if args.dataset == 'MNIST':
        args.model_path = './model/MNIST.pth'
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        _ = torch.manual_seed(1234)

        testset = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        plaintext_net = LeNet()
        plaintext_net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        plaintext_net.eval()

        hook = sy.TorchHook(torch)
        sam = sy.VirtualWorker(hook, id="sam")
        kelly = sy.VirtualWorker(hook, id="kelly")
        workers = [sam, kelly]
        crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")
        from syft.serde.compression import NO_COMPRESSION
        sy.serde.compression.default_compress_scheme = NO_COMPRESSION

        if args.test_only:
            file_name = './results/MNIST/precision_bit.txt'
            precision_bits = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
            for bit in precision_bits:
                print('current bit', bit)
                encryption_kwargs = dict(
                    workers=workers, 
                    crypto_provider=crypto_provider, 
                    protocol="snn", 
                    requires_grad=False,
                    base=2,
                    precision_fractional=bit,
                )
                new_plaintext_net = LeNet()
                new_plaintext_net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
                new_plaintext_net.eval()

                ptr_model = new_plaintext_net.send(sam)
                encrypted_model = ptr_model.encrypt(**encryption_kwargs).get()
                mpc_fuzzer = fuzzer(plaintext_net, encrypted_model, encryption_kwargs, workers, args)
                mpc_fuzzer.test_mpc(testloader)
        else:
            encryption_kwargs = dict(
                workers=workers, 
                crypto_provider=crypto_provider, 
                protocol="snn", 
                requires_grad=False,
                base=2,
                precision_fractional=16,
            )

            new_plaintext_net = LeNet()
            new_plaintext_net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
            new_plaintext_net.eval()

            ptr_model = new_plaintext_net.send(sam)
            encrypted_model = ptr_model.encrypt(**encryption_kwargs).get()

            mpc_fuzzer = fuzzer(plaintext_net, encrypted_model, encryption_kwargs, workers, args)
            mpc_fuzzer.start(testloader)
    elif args.dataset == 'Credit':
        args.model_path = './model/Credit.pth'
        data_list = np.load('./dataset/Credit/credit_card_clients_data.npy')
        label_list = np.load('./dataset/Credit/credit_card_clients_label.npy')
        data_tensor = torch.from_numpy(np.array(data_list))
        label_tensor = torch.from_numpy(np.array(label_list)).type(torch.LongTensor)
        test_data = data_tensor[-args.pool_size:].reshape((-1, args.batch_size, 23))
        test_label = label_tensor[-args.pool_size:].reshape((-1, args.batch_size))
        
        testloader = []
        for i in range(test_data.shape[0]):
            testloader.append((test_data[i], test_label[i]))

        _ = torch.manual_seed(1234)

        plaintext_net = Logistic()
        plaintext_net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        plaintext_net.eval()

        hook = sy.TorchHook(torch)
        sam = sy.VirtualWorker(hook, id="sam")
        kelly = sy.VirtualWorker(hook, id="kelly")
        workers = [sam, kelly]
        crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")
        from syft.serde.compression import NO_COMPRESSION
        sy.serde.compression.default_compress_scheme = NO_COMPRESSION

        if args.test_only:
            file_name = './results/Credit/precision_bit.txt'
            precision_bits = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
            for bit in precision_bits:
                print('current bit', bit)
                encryption_kwargs = dict(
                    workers=workers, 
                    crypto_provider=crypto_provider, 
                    protocol="snn", 
                    requires_grad=False,
                    base=2,
                    precision_fractional=bit,
                )
                new_plaintext_net = Logistic()
                new_plaintext_net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
                new_plaintext_net.eval()

                ptr_model = new_plaintext_net.send(sam)
                encrypted_model = ptr_model.encrypt(**encryption_kwargs).get()
                mpc_fuzzer = fuzzer(plaintext_net, encrypted_model, encryption_kwargs, workers, args)
                mpc_fuzzer.test_mpc(testloader)
        else:
            encryption_kwargs = dict(
                workers=workers, 
                crypto_provider=crypto_provider, 
                protocol="snn", 
                requires_grad=False,
                base=2,
                precision_fractional=16,
            )

            new_plaintext_net = Logistic()
            new_plaintext_net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
            new_plaintext_net.eval()

            ptr_model = new_plaintext_net.send(sam)
            encrypted_model = ptr_model.encrypt(**encryption_kwargs).get()

            mpc_fuzzer = fuzzer(plaintext_net, encrypted_model, encryption_kwargs, workers, args)
            mpc_fuzzer.start(testloader)

    elif args.dataset == 'Bank':
        args.model_path = './model/Bank_tfe.pth'
        data_list = np.load('./dataset/Bank/bank_data.npy')
        label_list = np.load('./dataset/Bank/bank_label.npy')
        data_tensor = torch.from_numpy(np.array(data_list))
        label_tensor = torch.from_numpy(np.array(label_list)).type(torch.LongTensor)
        test_data = data_tensor[-args.pool_size:].reshape((-1, args.batch_size, 20))
        test_label = label_tensor[-args.pool_size:].reshape((-1, args.batch_size))
        
        testloader = []
        for i in range(test_data.shape[0]):
            testloader.append((test_data[i], test_label[i]))

        _ = torch.manual_seed(1234)

        plaintext_net = BankNet_torch()
        plaintext_net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        plaintext_net.eval()

        hook = sy.TorchHook(torch)
        sam = sy.VirtualWorker(hook, id="sam")
        kelly = sy.VirtualWorker(hook, id="kelly")
        workers = [sam, kelly]
        crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")
        from syft.serde.compression import NO_COMPRESSION
        sy.serde.compression.default_compress_scheme = NO_COMPRESSION

        if args.test_only:
            file_name = './results/Bank/precision_bit.txt'
            precision_bits = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
            for bit in precision_bits:
                print('current bit', bit)
                encryption_kwargs = dict(
                    workers=workers, 
                    crypto_provider=crypto_provider, 
                    protocol="snn", 
                    requires_grad=False,
                    base=2,
                    precision_fractional=bit,
                )
                new_plaintext_net = BankNet_syft()
                new_plaintext_net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
                new_plaintext_net.eval()

                ptr_model = new_plaintext_net.send(sam)
                encrypted_model = ptr_model.encrypt(**encryption_kwargs).get()
                mpc_fuzzer = fuzzer(plaintext_net, encrypted_model, encryption_kwargs, workers, args)
                mpc_fuzzer.test_mpc(testloader)
        else:
            encryption_kwargs = dict(
                workers=workers, 
                crypto_provider=crypto_provider, 
                protocol="snn", 
                requires_grad=False,
                base=2,
                precision_fractional=12,
            )

            new_plaintext_net = BankNet_syft()
            new_plaintext_net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
            new_plaintext_net.eval()

            ptr_model = new_plaintext_net.send(sam)
            encrypted_model = ptr_model.encrypt(**encryption_kwargs).get()

            mpc_fuzzer = fuzzer(plaintext_net, encrypted_model, encryption_kwargs, workers, args)
            mpc_fuzzer.start(testloader)