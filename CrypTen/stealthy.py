import torch
import glob
import numpy as np
from PIL import Image
import argparse
import random, os
import torchvision
import torchvision.transforms as transforms


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
    inputs = table_to_tensor('./results/Bank/Fuzz_AE_*.pt')

    data_list = np.load('./dataset/Bank/bank_data.npy')
    label_list = np.load('./dataset/Bank/bank_label.npy')
    tests = data_list
    Avg = 0
    Base = 0
    index = []
    dis = []
    baseline = []
    for i in range(len(inputs)):
        temp_dis = np.inf
        temp_index = None
        for j in range(len(tests)):
            if np.linalg.norm(inputs[i] - tests[j]) < temp_dis:
                temp_dis = np.linalg.norm(inputs[i] - tests[j])
                temp_index = j
        index.append(temp_index)
        dis.append(temp_dis)
        baseline.append(np.linalg.norm(tests[temp_index]))
    print(inputs[0])
    print(np.mean(dis) / (tests[0].size), np.mean(baseline) / (tests[0].size))
