if __name__ == '__main__':
    import os
    import sys
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    p = os.path.abspath('../..')
    if p not in sys.path:
        sys.path.append(p)
    import warnings
    warnings.filterwarnings("ignore")

import logging, tqdm

import tensorflow as tf
import tensorflow.keras as keras

import tf_encrypted as tfe
from convert import decode

import numpy as np
import torch
from tf2torch import LeNet
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import copy, argparse, time

session_target = None

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

class PredictionClient:
    BATCH_SIZE = 1
    def __init__(self, player_name, local_data_file):
        self.player_name = player_name
        self.local_data_file = local_data_file

    def public_inputs(self, args):
        if args.dataset == 'MNIST':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            testset = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
            features_np = []
            labels_np = []
            for feature, label in testloader:
                feature = torch.transpose(feature, 1, 3)
                feature = torch.transpose(feature, 1, 2)
                feature = feature.clone().numpy()
                label = label.clone().numpy()
                features_np.append(feature)
                labels_np.append(label)
            return features_np, labels_np
        elif args.dataset == 'Credit':
            data_list = np.load('./dataset/Credit/credit_card_clients_data.npy')
            label_list = np.load('./dataset/Credit/credit_card_clients_label.npy')
            data_tensor = torch.from_numpy(np.array(data_list))
            label_tensor = torch.from_numpy(np.array(label_list)).type(torch.LongTensor)
            test_data = data_tensor[-args.pool_size:].reshape((-1, 1, 23))
            test_label = label_tensor[-args.pool_size:].reshape((-1, 1))
            features_np = []
            labels_np = []
            for i in range(test_data.shape[0]):
                features_np.append(test_data[i].clone().numpy())
                labels_np.append(test_label[i].clone().numpy())
            return features_np, labels_np
        elif args.dataset == 'Bank':
            data_list = np.load('./dataset/Bank/bank_data.npy')
            label_list = np.load('./dataset/Bank/bank_label.npy')
            data_tensor = torch.from_numpy(data_list)
            label_tensor = torch.from_numpy(label_list).type(torch.LongTensor)
            test_data = data_tensor[-args.pool_size:].reshape((-1, 1, 20))
            test_label = label_tensor[-args.pool_size:].reshape((-1, 1))
            features_np = []
            labels_np = []
            for i in range(test_data.shape[0]):
                features_np.append(test_data[i].clone().numpy())
                labels_np.append(test_label[i].clone().numpy())
            return features_np, labels_np
            

    @tfe.local_computation
    def provide_input(self, prediction_input):
        """Prepare input data for prediction."""
        return prediction_input

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
        elif args.dataset == 'Credit':
            self.mutation_scale = 0.01
            self.mutation_th = 30
        elif args.dataset == 'Bank':
            self.mutation_scale = 0.1
            self.mutation_th = 30
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.prediction_client = PredictionClient(player_name="prediction-client", local_data_file=None)
        self.features_np, self.labels_np = self.prediction_client.public_inputs(args)
        if args.dataset == 'MNIST':
            self.model_torch = LeNet()
            self.model_torch.load_state_dict(torch.load('../../model/MNIST_LeNet.pth', map_location=torch.device('cpu')))
        elif args.dataset == 'Credit':
            self.model_torch = Logistic()
            self.model_torch.load_state_dict(torch.load('../../model/Credit.pth', map_location=torch.device('cpu')))
        elif args.dataset == 'Bank':
            self.model_torch = BankNet()
            self.model_torch.load_state_dict(torch.load('../../model/Bank.pth', map_location=torch.device('cpu')))
        self.model_torch.eval()

    def initialize(self, args):
        if args.dataset == 'MNIST':
            with tfe.protocol.SecureNN():
                x_tf = tf.placeholder(dtype=tf.float32, shape=(1, 28, 28, 1))
                x = self.prediction_client.provide_input(x_tf)
                model = tfe.keras.Sequential([
                    tfe.keras.layers.Conv2D(6, 5, padding='same', batch_input_shape=(1, 28, 28, 1)),
                    tfe.keras.layers.BatchNormalization(),
                    tfe.keras.layers.Activation('sigmoid'),
                    tfe.keras.layers.AveragePooling2D(2, 2),
                    tfe.keras.layers.Conv2D(16, 5, padding='valid',),
                    tfe.keras.layers.BatchNormalization(),
                    tfe.keras.layers.Activation('sigmoid'),
                    tfe.keras.layers.AveragePooling2D(2, 2),
                    tfe.keras.layers.Flatten(),
                    tfe.keras.layers.Dense(120),
                    tfe.keras.layers.BatchNormalization(axis=1),
                    tfe.keras.layers.Activation('sigmoid'),
                    tfe.keras.layers.Dense(84),
                    tfe.keras.layers.BatchNormalization(axis=1),
                    tfe.keras.layers.Activation('sigmoid'),
                    tfe.keras.layers.Dense(10),])
                logits = model(x)
            plaintext_model = keras.models.load_model('../../model/MNIST_LeNet_Sigmoid.h5')
        elif args.dataset == 'Credit':
            with tfe.protocol.SecureNN():
                x_tf = tf.placeholder(dtype=tf.float32, shape=(1, 23))
                x = self.prediction_client.provide_input(x_tf)
                model = tfe.keras.Sequential([
                    tfe.keras.layers.Dense(120, batch_input_shape=(1, 23)),
                    tfe.keras.layers.Activation('sigmoid'),
                    tfe.keras.layers.Dense(2),])
                logits = model(x)
            plaintext_model = keras.models.load_model('../../model/Credit.h5')
        elif args.dataset == 'Bank':
            with tfe.protocol.SecureNN():
                x_tf = tf.placeholder(dtype=tf.float32, shape=(1, 20))
                x = self.prediction_client.provide_input(x_tf)
                model = tfe.keras.Sequential([
                    tfe.keras.layers.Dense(250, batch_input_shape=(1, 20)),
                    tfe.keras.layers.Activation('gelu'),
                    tfe.keras.layers.Dense(2),])
                logits = model(x)
            plaintext_model = keras.models.load_model('../../model/Bank.h5')
        weights = plaintext_model.get_weights()
        count = 0
        pbar = tqdm.tqdm(total=args.pool_size)
        plaintext_acc = 0
        crypten_acc = 0
        with tfe.Session(target=session_target) as sess:
            sess.run(tf.global_variables_initializer(), tag="init")
            model.set_weights(weights, sess)

            for i in range(len(self.features_np)):
                current_x = self.features_np[i]
                real_labels = self.labels_np[i]
                temp_mpc_label = []
                for k in range(10):
                    logits_mpc = sess.run(logits.reveal().to_native(), feed_dict={x_tf: current_x})
                    temp_mpc_label.append(np.argmax(logits_mpc, axis=1))
                label_mpc = max(temp_mpc_label, key=temp_mpc_label.count)
                count += 1

                a, b = plaintext_pred(current_x, self.model_torch, self.args)
                if b == real_labels:
                    plaintext_acc += 1
                if label_mpc == real_labels:
                    crypten_acc += 1
                if label_mpc != b and b == real_labels:
                    self.AEs.append(copy.deepcopy(current_x).astype(np.float32))
                    saving(current_x, len(self.AEs), self.args)
                else:
                    current_err = torch.norm(a - torch.from_numpy(logits_mpc))
                    self.pool.append(copy.deepcopy(current_x).astype(np.float32))
                    self.errs.append(copy.deepcopy(current_err))
                    self.true_labels.append(real_labels.copy())
                    self.mutation_times.append(1)
                pbar.update(1)
                pbar.set_postfix({'AEs': len(self.AEs), 'Pool': len(self.pool)})
                if count >= args.pool_size:
                    break
        print('Plaintext ACC:', plaintext_acc / count, 'TF-Encrypted ACC:', crypten_acc / count)
        self.original_inputs = copy.deepcopy(self.pool)

    def test_mpc(self, args):
        if args.dataset == 'MNIST':
            with tfe.protocol.SecureNN():
                x_tf = tf.placeholder(dtype=tf.float32, shape=(1, 28, 28, 1))
                x = self.prediction_client.provide_input(x_tf)
                model = tfe.keras.Sequential([
                    tfe.keras.layers.Conv2D(6, 5, padding='same', batch_input_shape=(1, 28, 28, 1)),
                    tfe.keras.layers.BatchNormalization(),
                    tfe.keras.layers.Activation('sigmoid'),
                    tfe.keras.layers.AveragePooling2D(2, 2),
                    tfe.keras.layers.Conv2D(16, 5, padding='valid',),
                    tfe.keras.layers.BatchNormalization(),
                    tfe.keras.layers.Activation('sigmoid'),
                    tfe.keras.layers.AveragePooling2D(2, 2),
                    tfe.keras.layers.Flatten(),
                    tfe.keras.layers.Dense(120),
                    tfe.keras.layers.BatchNormalization(axis=1),
                    tfe.keras.layers.Activation('sigmoid'),
                    tfe.keras.layers.Dense(84),
                    tfe.keras.layers.BatchNormalization(axis=1),
                    tfe.keras.layers.Activation('sigmoid'),
                    tfe.keras.layers.Dense(10),])
                logits = model(x)
            plaintext_model = keras.models.load_model('../../model/MNIST_LeNet_Sigmoid.h5')
        elif args.dataset == 'Credit':
            with tfe.protocol.SecureNN():
                x_tf = tf.placeholder(dtype=tf.float32, shape=(1, 23))
                x = self.prediction_client.provide_input(x_tf)
                model = tfe.keras.Sequential([
                    tfe.keras.layers.Dense(120, batch_input_shape=(1, 23)),
                    tfe.keras.layers.Activation('sigmoid'),
                    tfe.keras.layers.Dense(2),])
                logits = model(x)
            plaintext_model = keras.models.load_model('../../model/Credit.h5')
        elif args.dataset == 'Bank':
            with tfe.protocol.SecureNN():
                x_tf = tf.placeholder(dtype=tf.float32, shape=(1, 20))
                x = self.prediction_client.provide_input(x_tf)
                model = tfe.keras.Sequential([
                    tfe.keras.layers.Dense(250, batch_input_shape=(1, 20)),
                    tfe.keras.layers.Activation('gelu'),
                    tfe.keras.layers.Dense(2),])
                logits = model(x)
            plaintext_model = keras.models.load_model('../../model/Bank.h5')
        weights = plaintext_model.get_weights()
        count = 0
        pbar = tqdm.tqdm(total=args.pool_size)
        crypten_acc = 0
        with tfe.Session(target=session_target) as sess:
            sess.run(tf.global_variables_initializer(), tag="init")
            model.set_weights(weights, sess)
            for i in range(len(self.features_np)):
                current_x = self.features_np[i]
                real_labels = self.labels_np[i]
                temp_mpc_label = []
                for k in range(10):
                    logits_mpc = sess.run(logits.reveal().to_native(), feed_dict={x_tf: current_x})
                    temp_mpc_label.append(np.argmax(logits_mpc, axis=1))
                label_mpc = max(temp_mpc_label, key=temp_mpc_label.count)
                count += 1
                if label_mpc == real_labels:
                    crypten_acc += 1
                pbar.update(1)
                pbar.set_postfix({'ACC': crypten_acc / count})
                if count >= args.pool_size:
                    break
        print('TF-Encrypted ACC:', crypten_acc / count)
        return crypten_acc / count

    def mutation(self, seed_X, index):
        self.mutation_times[index] += 1
        return self.projection(seed_X + np.random.normal(0, self.mutation_scale, size=seed_X.shape))

    def projection(self, inputs, predefined_min=-0.4242, predefined_max=2.8214):
        if self.args.dataset == 'MNIST':
            inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min()) * (predefined_max - predefined_min) + predefined_min
            inputs = inputs.astype(np.float32)
        elif self.args.dataset == 'Credit' or self.args.dataset == 'Bank':
            inputs = np.clip(inputs, 0.0, 1.0).astype(np.float32)
        return inputs

    def getseed(self):
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

    def start(self, args):
        _path = './results/' + self.args.dataset + '/'
        self.initialize(args)
        if args.repaired:
            file_name = _path + 'repaired_fuzz_results.txt'
        else:
            file_name = _path + 'fuzz_results.txt'
        fuzz_start_time = time.time()
        fuzz_end_time = time.time()
        pbar = tqdm.tqdm(total=15000)
        if args.dataset == 'MNIST':
            with tfe.protocol.SecureNN():
                x_tf = tf.placeholder(dtype=tf.float32, shape=(1, 28, 28, 1))
                x = self.prediction_client.provide_input(x_tf)
                model = tfe.keras.Sequential([
                    tfe.keras.layers.Conv2D(6, 5, padding='same', batch_input_shape=(1, 28, 28, 1)),
                    tfe.keras.layers.BatchNormalization(),
                    tfe.keras.layers.Activation('sigmoid'),
                    tfe.keras.layers.AveragePooling2D(2, 2),
                    tfe.keras.layers.Conv2D(16, 5, padding='valid',),
                    tfe.keras.layers.BatchNormalization(),
                    tfe.keras.layers.Activation('sigmoid'),
                    tfe.keras.layers.AveragePooling2D(2, 2),
                    tfe.keras.layers.Flatten(),
                    tfe.keras.layers.Dense(120),
                    tfe.keras.layers.BatchNormalization(axis=1),
                    tfe.keras.layers.Activation('sigmoid'),
                    tfe.keras.layers.Dense(84),
                    tfe.keras.layers.BatchNormalization(axis=1),
                    tfe.keras.layers.Activation('sigmoid'),
                    tfe.keras.layers.Dense(10),])
                logits = model(x)
            plaintext_model = keras.models.load_model('../../model/MNIST_LeNet_Sigmoid.h5')
        elif args.dataset == 'Credit':
            with tfe.protocol.SecureNN():
                x_tf = tf.placeholder(dtype=tf.float32, shape=(1, 23))
                x = self.prediction_client.provide_input(x_tf)
                model = tfe.keras.Sequential([
                    tfe.keras.layers.Dense(120, batch_input_shape=(1, 23)),
                    tfe.keras.layers.Activation('sigmoid'),
                    tfe.keras.layers.Dense(2),])
                logits = model(x)
            plaintext_model = keras.models.load_model('../../model/Credit.h5')
        elif args.dataset == 'Bank':
            with tfe.protocol.SecureNN():
                x_tf = tf.placeholder(dtype=tf.float32, shape=(1, 20))
                x = self.prediction_client.provide_input(x_tf)
                model = tfe.keras.Sequential([
                    tfe.keras.layers.Dense(250, batch_input_shape=(1, 20)),
                    tfe.keras.layers.Activation('gelu'),
                    tfe.keras.layers.Dense(2),])
                logits = model(x)
            plaintext_model = keras.models.load_model('../../model/Bank.h5')

        weights = plaintext_model.get_weights()
        mutation_num = 0
        with tfe.Session(target=session_target) as sess:
            sess.run(tf.global_variables_initializer(), tag="init")
            model.set_weights(weights, sess)

            while mutation_num < 15000 and len(self.pool) > 0:
                current_x, current_err, true_label, index = self.getseed()
                current_x = self.mutation(current_x, index)
                mutation_num += 1
                a, b = plaintext_pred(current_x, self.model_torch, self.args)

                temp_mpc_label = []
                for k in range(10):
                    logits_mpc = sess.run(logits.reveal().to_native(), feed_dict={x_tf: current_x})
                    temp_mpc_label.append(np.argmax(logits_mpc, axis=1))
                label_mpc = max(temp_mpc_label, key=temp_mpc_label.count)

                if b != label_mpc and b == true_label:
                    self.AEs.append(current_x)
                    saving(current_x, len(self.AEs), self.args, original=False)
                    saving(self.original_inputs[index], len(self.AEs), self.args, original=True)
                    self.pop(index)
                    with open(file_name, 'a') as txt_file:
                        txt_file.write('%d, \t%f, %d\n'%(len(self.AEs), time.time() - fuzz_start_time, mutation_num))
                else:
                    new_err = torch.norm(a - torch.from_numpy(logits_mpc))
                    self.update(current_x, new_err, index)
                pbar.update(1)
                fuzz_end_time = time.time()
                pbar.set_postfix({'AEs': len(self.AEs), 'Pool': len(self.pool), 'Mutation': mutation_num})


def plaintext_pred(inputs, model_plaintext, args):
    if args.dataset == 'MNIST':
        inputs = np.transpose(inputs, (0, 3, 1, 2))
        inputs = torch.from_numpy(inputs)
        with torch.no_grad():
            outputs = model_plaintext(inputs)
            predicted = torch.argmax(outputs, dim=1).numpy()
            return outputs, predicted
    elif args.dataset == 'Credit' or args.dataset == 'Bank':
        inputs = torch.from_numpy(inputs)
        with torch.no_grad():
            outputs = model_plaintext(inputs)
            predicted = torch.argmax(outputs, dim=1).numpy()
            return outputs, predicted

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
    elif args.dataset == 'MNIST':
        img_name = img_name + '.jpg'
        image = np.transpose(image, (0, 3, 1, 2))
        image = torch.from_numpy(image)
        image = image * 0.3081 + 0.1307
        save_image(image, img_name)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pool_size', type=int, default=1000)
    parser.add_argument('--guide', type=str, default='err')
    parser.add_argument('--dataset', type=str, default='Credit')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--repaired', action='store_true')
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    device = "cpu"
    args.device = device
    if args.test_only:
        mpc_fuzzer = fuzzer(args)
        mpc_fuzzer.test_mpc(args)
    else:
        mpc_fuzzer = fuzzer(args)
        mpc_fuzzer.start(args)
