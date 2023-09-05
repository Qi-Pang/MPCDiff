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
from PIL import Image
import glob
import time, logging, tqdm, warnings, pickle
import multiprocessing

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
        self.internal1 = x.clone().detach()
        x = self.pool(self.activation(self.batchnorm2(self.conv2(x))))
        self.internal2 = x.clone().detach()
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        self.internal3 = x.clone().detach()
        x = self.activation(self.fc2(x))
        self.internal4 = x.clone().detach()
        x = self.fc3(x)
        return x

class LeNet_internal1(nn.Sequential):
    def __init__(self):
        super(LeNet_internal1, self).__init__()
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
        self.internal1 = x.clone().detach()
        return x

class LeNet_internal2(nn.Sequential):
    def __init__(self):
        super(LeNet_internal2, self).__init__()
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
        self.internal1 = x.clone().detach()
        x = self.pool(self.activation(self.batchnorm2(self.conv2(x))))
        self.internal2 = x.clone().detach()
        return x

class LeNet_internal3(nn.Sequential):
    
    def __init__(self):
        super(LeNet_internal3, self).__init__()
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
        self.internal1 = x.clone().detach()
        x = self.pool(self.activation(self.batchnorm2(self.conv2(x))))
        self.internal2 = x.clone().detach()
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        self.internal3 = x.clone().detach()
        return x

class LeNet_internal4(nn.Sequential):
    
    def __init__(self):
        super(LeNet_internal4, self).__init__()
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
        self.internal1 = x.clone().detach()
        x = self.pool(self.activation(self.batchnorm2(self.conv2(x))))
        self.internal2 = x.clone().detach()
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        self.internal3 = x.clone().detach()
        x = self.activation(self.fc2(x))
        self.internal4 = x.clone().detach()
        return x

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

class BankNet_torch(nn.Sequential):
    def __init__(self):
        super(BankNet_torch, self).__init__()
        self.fc1 = nn.Linear(20, 250)
        self.fc2 = nn.Linear(250, 2)
        self.activation = nn.GELU()
        self.internal1 = None

    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        self.internal1 = x.clone().detach()
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

class BankNet_syft_internal1(nn.Sequential):
    def __init__(self):
        super(BankNet_syft_internal1, self).__init__()
        self.fc1 = nn.Linear(20, 250)
        self.fc2 = nn.Linear(250, 2)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return x

def stat_internal_layer1(inputs, plaintext_net, workers, enc_params, model_enc, args):
    images = inputs[0:1]
    images = images.to(args.device)
    outputs_plaintext = plaintext_net(images)
    internal1_plaintext = plaintext_net.internal1[0]
    stats_max = np.zeros(internal1_plaintext.flatten().shape)
    store_index_max = []
    stats_min = np.zeros(internal1_plaintext.flatten().shape)
    store_index_min = []

    pbar = tqdm.tqdm(total=inputs.shape[0])
    batch_num = inputs.shape[0] // args.batch_size
    if inputs.shape[0] % args.batch_size:
        batch_num += 1

    for i in range(batch_num):
        begin_index = i * args.batch_size
        end_index = min(begin_index + args.batch_size, inputs.shape[0])
        images = inputs[begin_index:end_index]

        images = images.to(args.device)
        outputs_plaintext = plaintext_net(images)

        images_to_send = images.clone().detach()
        ptr_batch = images_to_send.send(workers[0])
        encrypted_batch = ptr_batch.encrypt(**enc_params).get()
        encrypted_result = model_enc(encrypted_batch).decrypt()

        internal1_plaintext = plaintext_net.internal1
        internal1_mpc = encrypted_result
        err_batch = (internal1_plaintext - internal1_mpc).abs()
        for k in range(err_batch.shape[0]):
            err = err_batch[k]
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
        pbar.update(images.shape[0])
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

def stat_internal_layer2(inputs, plaintext_net, workers, enc_params, model_enc, args):
    images = inputs[0:1]
    images = images.to(args.device)
    outputs_plaintext = plaintext_net(images)
    internal2_plaintext = plaintext_net.internal2[0]
    stats_max = np.zeros(internal2_plaintext.flatten().shape)
    store_index_max = []
    stats_min = np.zeros(internal2_plaintext.flatten().shape)
    store_index_min = []

    pbar = tqdm.tqdm(total=inputs.shape[0])
    batch_num = inputs.shape[0] // args.batch_size
    if inputs.shape[0] % args.batch_size:
        batch_num += 1

    for i in range(batch_num):
        begin_index = i * args.batch_size
        end_index = min(begin_index + args.batch_size, inputs.shape[0])
        images = inputs[begin_index:end_index]

        images = images.to(args.device)
        outputs_plaintext = plaintext_net(images)

        images_to_send = images.clone().detach()
        ptr_batch = images_to_send.send(workers[0])
        encrypted_batch = ptr_batch.encrypt(**enc_params).get()
        encrypted_result = model_enc(encrypted_batch).decrypt()

        internal2_plaintext = plaintext_net.internal2
        internal2_mpc = encrypted_result
        err_batch = (internal2_plaintext - internal2_mpc).abs()
        for k in range(err_batch.shape[0]):
            err = err_batch[k]
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
        pbar.update(images.shape[0])
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

def stat_internal_layer3(inputs, plaintext_net, workers, enc_params, model_enc, args):
    images = inputs[0:1]
    images = images.to(args.device)
    outputs_plaintext = plaintext_net(images)
    internal3_plaintext = plaintext_net.internal3[0]
    stats_max = np.zeros(internal3_plaintext.flatten().shape)
    store_index_max = []
    stats_min = np.zeros(internal3_plaintext.flatten().shape)
    store_index_min = []

    pbar = tqdm.tqdm(total=inputs.shape[0])
    batch_num = inputs.shape[0] // args.batch_size
    if inputs.shape[0] % args.batch_size:
        batch_num += 1

    for i in range(batch_num):
        begin_index = i * args.batch_size
        end_index = min(begin_index + args.batch_size, inputs.shape[0])
        images = inputs[begin_index:end_index]

        images = images.to(args.device)
        outputs_plaintext = plaintext_net(images)

        images_to_send = images.clone().detach()
        ptr_batch = images_to_send.send(workers[0])
        encrypted_batch = ptr_batch.encrypt(**enc_params).get()
        encrypted_result = model_enc(encrypted_batch).decrypt()

        internal3_plaintext = plaintext_net.internal3
        internal3_mpc = encrypted_result
        err_batch = (internal3_plaintext - internal3_mpc).abs()
        for k in range(err_batch.shape[0]):
            err = err_batch[k]
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
        pbar.update(images.shape[0])
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

def stat_internal_layer4(inputs, plaintext_net, workers, enc_params, model_enc, args):
    images = inputs[0:1]
    images = images.to(args.device)
    outputs_plaintext = plaintext_net(images)
    internal4_plaintext = plaintext_net.internal4[0]
    stats_max = np.zeros(internal4_plaintext.flatten().shape)
    store_index_max = []
    stats_min = np.zeros(internal4_plaintext.flatten().shape)
    store_index_min = []

    pbar = tqdm.tqdm(total=inputs.shape[0])
    batch_num = inputs.shape[0] // args.batch_size
    if inputs.shape[0] % args.batch_size:
        batch_num += 1

    for i in range(batch_num):
        begin_index = i * args.batch_size
        end_index = min(begin_index + args.batch_size, inputs.shape[0])
        images = inputs[begin_index:end_index]

        images = images.to(args.device)
        outputs_plaintext = plaintext_net(images)

        images_to_send = images.clone().detach()
        ptr_batch = images_to_send.send(workers[0])
        encrypted_batch = ptr_batch.encrypt(**enc_params).get()
        encrypted_result = model_enc(encrypted_batch).decrypt()

        internal4_plaintext = plaintext_net.internal4
        internal4_mpc = encrypted_result
        err_batch = (internal4_plaintext - internal4_mpc).abs()
        for k in range(err_batch.shape[0]):
            err = err_batch[k]
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
        pbar.update(images.shape[0])
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

def compare_labels(inputs, plaintext_net, enc_params, model_enc, args):
    same_num = 0
    total_num = inputs.shape[0]
    pbar = tqdm.tqdm(total=inputs.shape[0])
    batch_num = inputs.shape[0] // args.batch_size
    if inputs.shape[0] % args.batch_size:
        batch_num += 1
    for i in range(batch_num):
        begin_index = i * args.batch_size
        end_index = min(begin_index + args.batch_size, inputs.shape[0])
        images = inputs[begin_index:end_index]

        images_to_send = images.clone().detach()
        ptr_batch = images_to_send.send(workers[0])
        encrypted_batch = ptr_batch.encrypt(**enc_params).get()
        encrypted_result = model_enc(encrypted_batch).decrypt()
        _, encrypted_pred = torch.max(encrypted_result.data, 1)

        images = images.to(args.device)
        outputs_plaintext = plaintext_net(images)

        _, predicted_plaintext = torch.max(outputs_plaintext.data, 1)

        predicted_mpc = encrypted_pred
        same_num += (predicted_mpc == predicted_plaintext).sum().item()
        pbar.update(images.shape[0])
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

def png_to_tensor(folder_path):
    inputs = []
    image_name = []
    for im_path in glob.glob(folder_path):
        image_name.append(im_path)
        im = np.asarray(Image.open(im_path).convert('L'))
        inputs.append(im)
    return np.array(inputs) / 255, image_name

def table_to_tensor(folder_path):
    inputs = []
    for tensor_path in glob.glob(folder_path):
        cur_tensor = torch.load(tensor_path, map_location=torch.device('cpu')).numpy()
        inputs.append(cur_tensor)
    inputs = np.array(inputs)
    return inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--aug_layer', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='Credit')
    parser.add_argument('--test_only', action='store_true')
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    device = "cpu"
    args.device = device

    if args.dataset == 'MNIST':
        args.model_path = './model/MNIST.pth'
        inputs, image_name = png_to_tensor('./results/MNIST/Fuzz_AE_*.jpg')
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
        args.model_path = './model/Bank_tfe.pth'
        inputs = table_to_tensor('./results/Bank/Fuzz_AE_*.pt')
        inputs = inputs.reshape(inputs.shape[0], 20)
        inputs = torch.from_numpy(inputs).type(torch.float)

        plaintext_model = BankNet_torch().to(args.device)
        plaintext_model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        plaintext_model.eval()

        mpc_model = BankNet_syft().to(args.device)
        mpc_model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        mpc_model.eval()

        mpc_model_internal1 = BankNet_syft_internal1().to(args.device)
        mpc_model_internal1.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        mpc_model_internal1.eval()
        _path = './results/Bank/'

    sy.register_counter(args)
    hook = sy.TorchHook(torch)
    sam = sy.VirtualWorker(hook, id="sam")
    kelly = sy.VirtualWorker(hook, id="kelly")
    workers = [sam, kelly]
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")
    from syft.serde.compression import NO_COMPRESSION
    sy.serde.compression.default_compress_scheme = NO_COMPRESSION

    encryption_kwargs = dict(
        workers=workers, 
        crypto_provider=crypto_provider, 
        protocol="snn", 
        requires_grad=False,
        base=2,
        precision_fractional=16,
    )

    if args.aug_layer == 1:
        ptr_model = mpc_model_internal1.send(sam)
        encrypted_model = ptr_model.encrypt(**encryption_kwargs).get()

        store_index_max, store_index_min = stat_internal_layer1(inputs, plaintext_model, workers, encryption_kwargs, encrypted_model, args)

        with open(_path + 'layer1_max.pkl', 'wb') as fp:
            pickle.dump(store_index_max, fp)
        with open(_path + 'layer1_min.pkl', 'wb') as fp:
            pickle.dump(store_index_min, fp)
        print('Voting length: %d, %d' % (len(store_index_max), len(store_index_min)))

    elif args.aug_layer == 2:
        ptr_model = mpc_model_internal2.send(sam)
        encrypted_model = ptr_model.encrypt(**encryption_kwargs).get()

        store_index_max, store_index_min = stat_internal_layer2(inputs, plaintext_model, workers, encryption_kwargs, encrypted_model, args)
        with open(_path + 'layer2_max.pkl', 'wb') as fp:
            pickle.dump(store_index_max, fp)
        with open(_path + 'layer2_min.pkl', 'wb') as fp:
            pickle.dump(store_index_min, fp)
        print('Voting length: %d, %d' % (len(store_index_max), len(store_index_min)))

    elif args.aug_layer == 3:
        ptr_model = mpc_model_internal3.send(sam)
        encrypted_model = ptr_model.encrypt(**encryption_kwargs).get()

        store_index_max, store_index_min = stat_internal_layer3(inputs, plaintext_model, workers, encryption_kwargs, encrypted_model, args)

        with open(_path + 'layer3_max.pkl', 'wb') as fp:
            pickle.dump(store_index_max, fp)
        with open(_path + 'layer3_min.pkl', 'wb') as fp:
            pickle.dump(store_index_min, fp)
        print('Voting length: %d, %d' % (len(store_index_max), len(store_index_min)))

    elif args.aug_layer == 4:
        ptr_model = mpc_model_internal4.send(sam)
        encrypted_model = ptr_model.encrypt(**encryption_kwargs).get()

        store_index_max, store_index_min = stat_internal_layer4(inputs, plaintext_model, workers, encryption_kwargs, encrypted_model, args)

        with open(_path + 'layer4_max.pkl', 'wb') as fp:
            pickle.dump(store_index_max, fp)
        with open(_path + 'layer4_min.pkl', 'wb') as fp:
            pickle.dump(store_index_min, fp)
        print('Voting length: %d, %d' % (len(store_index_max), len(store_index_min)))

    else:
        if args.test_only:
            precision_bits = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
            for bit in precision_bits:
                encryption_kwargs = dict(
                    workers=workers, 
                    crypto_provider=crypto_provider, 
                    protocol="snn", 
                    requires_grad=False,
                    base=2,
                    precision_fractional=bit,
                )
                print('current bit:', bit)
                if args.dataset == 'MNIST':
                    mpc_model = LeNet().to(args.device)
                    mpc_model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
                    mpc_model.eval()
                elif args.dataset == 'Credit':
                    mpc_model = Logistic().to(args.device)
                    mpc_model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
                    mpc_model.eval()
                elif args.dataset == 'Bank':
                    mpc_model = BankNet_syft().to(args.device)
                    mpc_model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
                    mpc_model.eval()
                ptr_model = mpc_model.send(sam)
                encrypted_model = ptr_model.encrypt(**encryption_kwargs).get()

                same_nums, total_nums = compare_labels(inputs, plaintext_model, encryption_kwargs, encrypted_model, args)
                print(same_nums, total_nums)
        else:
            ptr_model = mpc_model.send(sam)
            encrypted_model = ptr_model.encrypt(**encryption_kwargs).get()

            same_nums, total_nums = compare_labels(inputs, plaintext_model, encryption_kwargs, encrypted_model, args)
            print(same_nums, total_nums)

