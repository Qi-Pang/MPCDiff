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

from ast import arg
import glob, torch, tqdm, argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tf2torch import LeNet
import tf_encrypted as tfe
import pickle

session_target = None

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

class PredictionClient:
    BATCH_SIZE = 1
    def __init__(self, player_name, local_data_file):
        self.player_name = player_name
        self.local_data_file = local_data_file

    def public_inputs(self):
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

    @tfe.local_computation
    def provide_input(self, prediction_input):
        """Prepare input data for prediction."""
        return prediction_input

def png_to_tensor(folder_path):
    inputs = []
    image_name = []
    for im_path in glob.glob(folder_path):
        image_name.append(im_path)
        im = np.asarray(Image.open(im_path).convert('L'))
        inputs.append(im)
    inputs = np.array(inputs) / 255
    inputs = (inputs - 0.1307) / 0.3081
    return inputs, image_name

def table_to_tensor(folder_path):
    inputs = []
    for tensor_path in glob.glob(folder_path):
        cur_tensor = torch.load(tensor_path, map_location=torch.device('cpu'))
        inputs.append(cur_tensor)
    inputs = np.array(inputs)
    return inputs

def compare_labels(inputs, plaintext_net, prediction_client, args):
    if args.dataset == 'MNIST':
        with tfe.protocol.SecureNN():
            x_tf = tf.placeholder(dtype=tf.float32, shape=(1, 28, 28, 1))
            x = prediction_client.provide_input(x_tf)
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
        weights = keras.models.load_model('../../model/MNIST_LeNet_Sigmoid.h5').get_weights()
    elif args.dataset == 'Credit':
        with tfe.protocol.SecureNN():
            x_tf = tf.placeholder(dtype=tf.float32, shape=(1, 23))
            x = prediction_client.provide_input(x_tf)
            model = tfe.keras.Sequential([
                tfe.keras.layers.Dense(120, batch_input_shape=(1, 23)),
                tfe.keras.layers.Activation('sigmoid'),
                tfe.keras.layers.Dense(2),])
            logits = model(x)
        weights = keras.models.load_model('../../model/Credit.h5').get_weights()
    elif args.dataset == 'Bank':
        with tfe.protocol.SecureNN():
            x_tf = tf.placeholder(dtype=tf.float32, shape=(1, 20))
            x = prediction_client.provide_input(x_tf)
            model = tfe.keras.Sequential([
                tfe.keras.layers.Dense(250, batch_input_shape=(1, 20)),
                tfe.keras.layers.Activation('gelu'),
                tfe.keras.layers.Dense(2),])
            logits = model(x)
        weights = keras.models.load_model('../../model/Bank.h5').get_weights()


    num_total = 0
    num_same = 0
    with tfe.Session(target=session_target) as sess:
        sess.run(tf.global_variables_initializer(), tag="init")
        model.set_weights(weights, sess)
        for i in tqdm.tqdm(range(inputs.shape[0])):
            if args.dataset == 'MNIST':
                current_x = inputs[i].reshape(1, 28, 28, 1)
                temp_mpc_label = []
                for k in range(10):
                    logits_mpc = sess.run(logits.reveal().to_native(), feed_dict={x_tf: current_x})
                    temp_mpc_label.append(np.argmax(logits_mpc, axis=1))
                label_mpc = max(temp_mpc_label, key=temp_mpc_label.count)
                current_x = np.transpose(current_x, (0, 3, 1, 2))
            elif args.dataset == 'Credit':
                current_x = inputs[i].reshape(1, 23)
                temp_mpc_label = []
                for k in range(10):
                    logits_mpc = sess.run(logits.reveal().to_native(), feed_dict={x_tf: current_x})
                    temp_mpc_label.append(np.argmax(logits_mpc, axis=1))
                label_mpc = max(temp_mpc_label, key=temp_mpc_label.count)
            elif args.dataset == 'Bank':
                current_x = inputs[i].reshape(1, 20)
                temp_mpc_label = []
                for k in range(10):
                    logits_mpc = sess.run(logits.reveal().to_native(), feed_dict={x_tf: current_x})
                    temp_mpc_label.append(np.argmax(logits_mpc, axis=1))
                label_mpc = max(temp_mpc_label, key=temp_mpc_label.count)
            current_x = torch.from_numpy(current_x).type(torch.float)
            with torch.no_grad():
                logits_plaintext = plaintext_net(current_x)
                label_plaintext = torch.argmax(logits_plaintext, dim=1).numpy()
            if label_mpc == label_plaintext:
                num_same += 1
            num_total += 1
    return num_same, num_total

def stat_internal_layer1(inputs, plaintext_net, prediction_client, args):
    if args.dataset == 'MNIST':
        with tfe.protocol.SecureNN():
            x_tf = tf.placeholder(dtype=tf.float32, shape=(1, 28, 28, 1))
            x = prediction_client.provide_input(x_tf)
            model = tfe.keras.Sequential([
                tfe.keras.layers.Conv2D(6, 5, padding='same', batch_input_shape=(1, 28, 28, 1)),
                tfe.keras.layers.BatchNormalization(),
                tfe.keras.layers.Activation('sigmoid')])
            logits = model(x)
        weights = keras.models.load_model('../../model/MNIST_LeNet_Sigmoid.h5').get_weights()
    elif args.dataset == 'Credit':
        with tfe.protocol.SecureNN():
            x_tf = tf.placeholder(dtype=tf.float32, shape=(1, 23))
            x = prediction_client.provide_input(x_tf)
            model = tfe.keras.Sequential([
                tfe.keras.layers.Dense(120, batch_input_shape=(1, 23)),
                tfe.keras.layers.Activation('sigmoid')])
            logits = model(x)
        weights = keras.models.load_model('../../model/Credit.h5').get_weights()
    elif args.dataset == 'Bank':
        with tfe.protocol.SecureNN():
            x_tf = tf.placeholder(dtype=tf.float32, shape=(1, 20))
            x = prediction_client.provide_input(x_tf)
            model = tfe.keras.Sequential([
                tfe.keras.layers.Dense(250, batch_input_shape=(1, 20)),
                tfe.keras.layers.Activation('gelu')])
            logits = model(x)
        weights = keras.models.load_model('../../model/Bank.h5').get_weights()
    store_index_max = []
    store_index_min = []
    stats_max = None
    stats_min = None
    with tfe.Session(target=session_target) as sess:
        sess.run(tf.global_variables_initializer(), tag="init")
        model.set_weights(weights, sess)
        for i in tqdm.tqdm(range(inputs.shape[0])):
            if args.dataset == 'MNIST':
                current_x = inputs[i].reshape(1, 28, 28, 1)
                internal1_mpc = sess.run(logits.reveal().to_native(), feed_dict={x_tf: current_x})
                current_x = np.transpose(current_x, (0, 3, 1, 2))
            elif args.dataset == 'Credit':
                current_x = inputs[i].reshape(1, 23)
                internal1_mpc = sess.run(logits.reveal().to_native(), feed_dict={x_tf: current_x})
            elif args.dataset == 'Bank':
                current_x = inputs[i].reshape(1, 20)
                internal1_mpc = sess.run(logits.reveal().to_native(), feed_dict={x_tf: current_x})
            current_x = torch.from_numpy(current_x).type(torch.float)
            with torch.no_grad():
                logits_plaintext = plaintext_net(current_x)
                internal1_plaintext = plaintext_net.internal1.numpy()
                if args.dataset == 'MNIST':
                    internal1_plaintext = np.transpose(internal1_plaintext, (0, 2, 3, 1))
                elif args.dataset == 'Credit':
                    internal1_plaintext = np.transpose(internal1_plaintext, (0, 1))
                elif args.dataset == 'Bank':
                    internal1_plaintext = np.transpose(internal1_plaintext, (0, 1))
            if stats_max is None or stats_min is None:
                stats_max = np.zeros(internal1_plaintext.flatten().shape)
                stats_min = np.zeros(internal1_plaintext.flatten().shape)
            internal1_mpc = torch.from_numpy(internal1_mpc)
            internal1_plaintext = torch.from_numpy(internal1_plaintext)
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

def stat_internal_layer2(inputs, plaintext_net, prediction_client, args):
    with tfe.protocol.SecureNN():
        x_tf = tf.placeholder(dtype=tf.float32, shape=(1, 28, 28, 1))
        x = prediction_client.provide_input(x_tf)
        model = tfe.keras.Sequential([
            tfe.keras.layers.Conv2D(6, 5, padding='same', batch_input_shape=(1, 28, 28, 1)),
            tfe.keras.layers.BatchNormalization(),
            tfe.keras.layers.Activation('sigmoid'),
            tfe.keras.layers.AveragePooling2D(2, 2),
            tfe.keras.layers.Conv2D(16, 5, padding='valid',),
            tfe.keras.layers.BatchNormalization(),
            tfe.keras.layers.Activation('sigmoid')])
        logits = model(x)

    weights = keras.models.load_model('../../model/MNIST_LeNet_Sigmoid.h5').get_weights()
    store_index_max = []
    store_index_min = []
    stats_max = None
    stats_min = None
    with tfe.Session(target=session_target) as sess:
        sess.run(tf.global_variables_initializer(), tag="init")
        model.set_weights(weights, sess)
        for i in tqdm.tqdm(range(inputs.shape[0])):
            current_x = inputs[i].reshape(1, 28, 28, 1)
            internal2_mpc = sess.run(logits.reveal().to_native(), feed_dict={x_tf: current_x})
            current_x = np.transpose(current_x, (0, 3, 1, 2))
            current_x = torch.from_numpy(current_x).type(torch.float)
            with torch.no_grad():
                logits_plaintext = plaintext_net(current_x)
                internal2_plaintext = plaintext_net.internal2.numpy()
                internal2_plaintext = np.transpose(internal2_plaintext, (0, 2, 3, 1))
            if stats_max is None or stats_min is None:
                stats_max = np.zeros(internal2_plaintext.flatten().shape)
                stats_min = np.zeros(internal2_plaintext.flatten().shape)
            internal2_mpc = torch.from_numpy(internal2_mpc)
            internal2_plaintext = torch.from_numpy(internal2_plaintext)
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

def stat_internal_layer3(inputs, plaintext_net, prediction_client, args):
    with tfe.protocol.SecureNN():
        x_tf = tf.placeholder(dtype=tf.float32, shape=(1, 28, 28, 1))
        x = prediction_client.provide_input(x_tf)
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
            tfe.keras.layers.Activation('sigmoid')])
        logits = model(x)

    weights = keras.models.load_model('../../model/MNIST_LeNet_Sigmoid.h5').get_weights()
    store_index_max = []
    store_index_min = []
    stats_max = None
    stats_min = None
    with tfe.Session(target=session_target) as sess:
        sess.run(tf.global_variables_initializer(), tag="init")
        model.set_weights(weights, sess)
        for i in tqdm.tqdm(range(inputs.shape[0])):
            current_x = inputs[i].reshape(1, 28, 28, 1)
            internal3_mpc = sess.run(logits.reveal().to_native(), feed_dict={x_tf: current_x})
            current_x = np.transpose(current_x, (0, 3, 1, 2))
            current_x = torch.from_numpy(current_x).type(torch.float)
            with torch.no_grad():
                logits_plaintext = plaintext_net(current_x)
                internal3_plaintext = plaintext_net.internal3.numpy()
            if stats_max is None or stats_min is None:
                stats_max = np.zeros(internal3_plaintext.flatten().shape)
                stats_min = np.zeros(internal3_plaintext.flatten().shape)
            internal3_mpc = torch.from_numpy(internal3_mpc)
            internal3_plaintext = torch.from_numpy(internal3_plaintext)
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

def stat_internal_layer4(inputs, plaintext_net, prediction_client, args):
    with tfe.protocol.SecureNN():
        x_tf = tf.placeholder(dtype=tf.float32, shape=(1, 28, 28, 1))
        x = prediction_client.provide_input(x_tf)
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
            tfe.keras.layers.Activation('sigmoid')])
        logits = model(x)

    weights = keras.models.load_model('../../model/MNIST_LeNet_Sigmoid.h5').get_weights()
    store_index_max = []
    store_index_min = []
    stats_max = None
    stats_min = None
    with tfe.Session(target=session_target) as sess:
        sess.run(tf.global_variables_initializer(), tag="init")
        model.set_weights(weights, sess)
        for i in tqdm.tqdm(range(inputs.shape[0])):
            current_x = inputs[i].reshape(1, 28, 28, 1)
            internal4_mpc = sess.run(logits.reveal().to_native(), feed_dict={x_tf: current_x})
            current_x = np.transpose(current_x, (0, 3, 1, 2))
            current_x = torch.from_numpy(current_x).type(torch.float)
            with torch.no_grad():
                logits_plaintext = plaintext_net(current_x)
                internal4_plaintext = plaintext_net.internal4.numpy()
            if stats_max is None or stats_min is None:
                stats_max = np.zeros(internal4_plaintext.flatten().shape)
                stats_min = np.zeros(internal4_plaintext.flatten().shape)
            internal4_mpc = torch.from_numpy(internal4_mpc)
            internal4_plaintext = torch.from_numpy(internal4_plaintext)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Credit')
    parser.add_argument('--aug_layer', type=int, default=0)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    device = "cpu"
    args.device = device
    pred_client = PredictionClient(player_name="prediction-client", local_data_file=None)

    if args.dataset == 'MNIST':
        inputs, image_name = png_to_tensor('./results/LeNet/Fuzz_AE_*.jpg')
        model_torch = LeNet()
        model_torch.load_state_dict(torch.load('../../model/MNIST_LeNet.pth', map_location=torch.device('cpu')))
        model_torch.eval()
        _path = './results/MNIST/'
    elif args.dataset == 'Credit':
        inputs = table_to_tensor('./results/Credit/Fuzz_AE_*.pt')
        model_torch = Logistic()
        model_torch.load_state_dict(torch.load('../../model/Credit.pth', map_location=torch.device('cpu')))
        model_torch.eval()
        _path = './results/Credit/'
    elif args.dataset == 'Bank':
        inputs = table_to_tensor('./results/Bank/Fuzz_AE_*.pt')
        model_torch = BankNet()
        model_torch.load_state_dict(torch.load('../../model/Bank.pth', map_location=torch.device('cpu')))
        model_torch.eval()
        _path = './results/Bank/'

    if args.aug_layer == 0:
        num_same, num_total = compare_labels(inputs, model_torch, pred_client, args)
        print(num_same, num_total)
    elif args.aug_layer == 1:
        store_index_max, store_index_min = stat_internal_layer1(inputs, model_torch, pred_client, args)
        with open(_path + 'layer1_max.pkl', 'wb') as fp:
            pickle.dump(store_index_max, fp)
        with open(_path + 'layer1_min.pkl', 'wb') as fp:
            pickle.dump(store_index_min, fp)
        print('Voting length: %d, %d' % (len(store_index_max), len(store_index_min)))
    elif args.aug_layer == 2:
        store_index_max, store_index_min = stat_internal_layer2(inputs, model_torch, pred_client, args)
        with open('./results/LeNet/layer2_max.pkl', 'wb') as fp:
            pickle.dump(store_index_max, fp)
        with open('./results/LeNet/layer2_min.pkl', 'wb') as fp:
            pickle.dump(store_index_min, fp)
        print('Voting length: %d, %d' % (len(store_index_max), len(store_index_min)))
    elif args.aug_layer == 3:
        store_index_max, store_index_min = stat_internal_layer3(inputs, model_torch, pred_client, args)
        with open('./results/LeNet/layer3_max.pkl', 'wb') as fp:
            pickle.dump(store_index_max, fp)
        with open('./results/LeNet/layer3_min.pkl', 'wb') as fp:
            pickle.dump(store_index_min, fp)
        print('Voting length: %d, %d' % (len(store_index_max), len(store_index_min)))
    elif args.aug_layer == 4:
        store_index_max, store_index_min = stat_internal_layer4(inputs, model_torch, pred_client, args)
        with open('./results/LeNet/layer4_max.pkl', 'wb') as fp:
            pickle.dump(store_index_max, fp)
        with open('./results/LeNet/layer4_min.pkl', 'wb') as fp:
            pickle.dump(store_index_min, fp)
        print('Voting length: %d, %d' % (len(store_index_max), len(store_index_min)))
