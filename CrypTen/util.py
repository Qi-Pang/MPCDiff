import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import argparse
from train_plaintext import LeNet
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
torch.set_num_threads(1)

def evaluate(inputs, args):
    net = LeNet()
    net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    net.eval()
    ground_truth = []
    with torch.no_grad():
        for i in range(inputs.shape[0]):
            images = inputs[i:i+1]
            images = images.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            ground_truth.append(predicted.clone())
        return ground_truth

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
    for i in tqdm.tqdm(range(100)):
        inputCrypt = inputs[i:i+1]
        private_input = crypten.cryptensor(inputCrypt, src=1)
        OutputCrypt = private_model(private_input)
        OutputCrypt = OutputCrypt.get_plain_text()
        _, predicted = torch.max(OutputCrypt.data, 1)
        if predicted == labels[i]:
            correct_num += 1
    print(correct_num, len(inputs))


def png_to_tensor(folder_path):
    inputs = []
    image_name = []
    for im_path in glob.glob(folder_path):
        image_name.append(im_path)
        im = np.asarray(Image.open(im_path).convert('L'))
        inputs.append(im)
    return np.array(inputs) / 255, image_name

class augdata(data.Dataset):
    def __init__(self, input_data, input_label):
        self.data = input_data
        self.label = input_label
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch_num', type=int, default=25)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--iter', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model_path', type=str, default='./model/MNIST_LeNet_sigmoid_batchnorm.pth')
    parser.add_argument('--check_Crypten', default=False, action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    device = "cpu"
    args.device = device

    inputs, image_name = png_to_tensor('./results/*.png')
    inputs = torch.tensor(inputs)
    trans = transforms.Normalize((0.1307,), (0.3081,))
    inputs = trans(inputs).reshape(inputs.shape[0], 1, 28, 28)
    inputs = inputs.type(torch.float)
    print(inputs.shape, inputs.max(), inputs.min())
    print(image_name[0], image_name[1], image_name[2], image_name[3], image_name[4])
    labels = evaluate(inputs, args)
    crypten.init()
    model_counter.register_counter()
    verify_crypt(inputs, labels, args)
    print('computation time: ', model_counter.computation_time)