import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
import argparse, sys
import numpy as np
from examples.util import NoopContextManager
from examples.meters import AverageMeter
import crypten
import crypten.communicator as comm
import time, logging, tqdm, warnings, pickle
import crypten.mpc as mpc

def chebyshev_series(func, width, terms):
    r"""Computes Chebyshev coefficients
    For n = terms, the ith Chebyshev series coefficient is
    .. math::
        c_i = 2/n \sum_{k=1}^n \cos(j(2k-1)\pi / 4n) f(w\cos((2k-1)\pi / 4n))
    Args:
        func (function): function to be approximated
        width (int): approximation will support inputs in range [-width, width]
        terms (int): number of Chebyshev terms used in approximation
    Returns:
        Chebyshev coefficients with shape equal to num of terms.
    """
    n_range = torch.arange(start=0, end=terms).float()
    x = width * torch.cos((n_range + 0.5) * np.pi / terms)
    y = func(x)
    cos_term = torch.cos(torch.ger(n_range, n_range + 0.5) * np.pi / terms)
    coeffs = (2 / terms) * torch.sum(y * cos_term, axis=1)
    return coeffs

def _chebyshev_polynomials(x, terms):
    x = x.flatten()
    polynomials = [x.clone()]
    y = 4 * x * x - 2
    z = y - 1
    polynomials.append(z * x)
    for k in range(2, terms//2):
        next_polynoimal = y * polynomials[k-1] - polynomials[k-2]
        polynomials.append(next_polynoimal)
    polynomials = torch.stack(polynomials)
    return polynomials

def exp(x):
    r"""Approximates the exponential function using a limit approximation:
    .. math::
        exp(x) = \lim_{n \\rightarrow \\infty} (1 + x / n) ^ n
    Here we compute exp by choosing n = 2 ** d for some large d equal to
    `iterations`. We then compute (1 + x / n) once and square `d` times.
    Set the number of iterations for the limit approximation with
    config.exp_iterations.
    """  # noqa: W605
    iters = 8
    result = 1 + x / (2**iters)
    for _ in range(iters):
        result = result**2
    return result

def inv_sqrt(x):
    r"""
    Computes the inverse square root of the input using the Newton-Raphson method.
    Configuration params:
        sqrt_nr_iters (int):  determines the number of Newton-Raphson iterations to run.
        sqrt_nr_initial (tensor): sets the initial value for the Newton-Raphson iterations.
                    By default, this will be set to allow the method to converge over a
                    fairly large domain.
    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
    """
    initial = None
    iters = 3

    if initial is None:
        y = exp(-(x / 2 + 0.2)) * 2.2 + 0.2 - x / 1024
    else:
        y = initial
    for _ in range(iters):
        y = y * (3 - x * y**2) / 2
    return y

def inv_sqrt_highAcc(x):
    r"""
    Computes the inverse square root of the input using the Newton-Raphson method.
    Configuration params:
        sqrt_nr_iters (int):  determines the number of Newton-Raphson iterations to run.
        sqrt_nr_initial (tensor): sets the initial value for the Newton-Raphson iterations.
                    By default, this will be set to allow the method to converge over a
                    fairly large domain.
    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
    """
    initial = None
    iters = 6

    if initial is None:
        y = exp(-(x / 2 + 0.2)) * 2.2 + 0.2 - x / 1024
    else:
        y = initial

    for _ in range(iters):
        y = y * (3 - x * y**2) / 2

    return y

def inv_sqrt_lowAcc(x):
    r"""
    Computes the inverse square root of the input using the Newton-Raphson method.
    Configuration params:
        sqrt_nr_iters (int):  determines the number of Newton-Raphson iterations to run.
        sqrt_nr_initial (tensor): sets the initial value for the Newton-Raphson iterations.
                    By default, this will be set to allow the method to converge over a
                    fairly large domain.
    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
    """
    initial = None
    iters = 3

    if initial is None:
        y = exp(-(x / 2 + 0.2)) * 2.2 + 0.2 - x / 1024
    else:
        y = initial
    for _ in range(iters):
        y = y * (3 - x * y**2) / 2
    return y

def mpcsigmoid(x):
    x = x / 2
    terms = 6
    coeffs = chebyshev_series(torch.tanh, 1, terms)[1::2]
    tanh_polys = _chebyshev_polynomials(x, terms)
    approtanh = torch.clamp(coeffs.matmul(tanh_polys), -1.0, 1.0)
    approtanh = approtanh.reshape(x.shape)
    return approtanh / 2 + 0.5

def mpcsigmoid_highAcc(x):
    x = x / 2
    terms = 6
    coeffs = chebyshev_series(torch.tanh, 1, terms)[1::2]
    tanh_polys = _chebyshev_polynomials(x, terms)
    approtanh = torch.clamp(coeffs.matmul(tanh_polys), -1.0, 1.0)
    approtanh = approtanh.reshape(x.shape)
    
    return approtanh / 2 + 0.5

def mpcbatchnorm2d(x, running_mean, running_var, weight, bias, eps):
    assert x.dim() == 4
    x = x.transpose(1, 3)
    norm_result = (x - running_mean) * inv_sqrt(running_var + eps) * weight + bias
    return norm_result.transpose(1, 3)

def mpcbatchnorm2d_highAcc(x, running_mean, running_var, weight, bias, eps):
    assert x.dim() == 4
    x = x.transpose(1, 3)
    norm_result = (x - running_mean) * inv_sqrt_highAcc(running_var + eps) * weight + bias
    return norm_result.transpose(1, 3)

def mpcbatchnorm2d_lowAcc(x, running_mean, running_var, weight, bias, eps):
    assert x.dim() == 4
    x = x.transpose(1, 3)
    norm_result = (x - running_mean) * inv_sqrt_lowAcc(running_var + eps) * weight + bias
    return norm_result.transpose(1, 3)

def mpcbatchnorm1d(x, running_mean, running_var, weight, bias, eps):
    assert x.dim() == 2 or x.dim() == 3
    norm_result = (x - running_mean) * inv_sqrt(running_var + eps) * weight + bias
    # norm_result = norm_result.reshape(x_shape)
    return norm_result

def mpcbatchnorm1d_highAcc(x, running_mean, running_var, weight, bias, eps):
    assert x.dim() == 2 or x.dim() == 3
    norm_result = (x - running_mean) * inv_sqrt_highAcc(running_var + eps) * weight + bias
    return norm_result

class LeNet(nn.Sequential):
    
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.batchnorm3 = nn.BatchNorm1d(120)
        self.batchnorm4 = nn.BatchNorm1d(84)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.internal1 = None
        self.internal2 = None
        self.internal3 = None
        self.internal4 = None

    
    def forward(self, x):
        x = F.sigmoid(self.batchnorm1(self.conv1(x)))
        self.internal1 = x.clone()
        x = self.pool(x)
        x = F.sigmoid(self.batchnorm2(self.conv2(x)))
        self.internal2 = x.clone()
        x = self.pool(x)
        x = x.view(-1, 5*5*16)
        x = F.sigmoid(self.batchnorm3(self.fc1(x)))
        self.internal3 = x.clone()
        x = F.sigmoid(self.batchnorm4(self.fc2(x)))
        self.internal4 = x.clone()
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
        self.batchnorm3 = nn.BatchNorm1d(120)
        self.batchnorm4 = nn.BatchNorm1d(84)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.internal1 = None
        self.internal2 = None
        self.internal3 = None
        self.internal4 = None

    
    def forward(self, x):
        x = F.sigmoid(self.batchnorm1(self.conv1(x)))
        self.internal1 = x.clone()
        return self.internal1

class LeNet_internal2(nn.Sequential):
    
    def __init__(self):
        super(LeNet_internal2, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.batchnorm3 = nn.BatchNorm1d(120)
        self.batchnorm4 = nn.BatchNorm1d(84)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.internal1 = None
        self.internal2 = None
        self.internal3 = None
        self.internal4 = None

    
    def forward(self, x):
        x = F.sigmoid(self.batchnorm1(self.conv1(x)))
        self.internal1 = x.clone()
        x = self.pool(x)
        x = F.sigmoid(self.batchnorm2(self.conv2(x)))
        self.internal2 = x.clone()
        return self.internal2

class LeNet_internal3(nn.Sequential):
    
    def __init__(self):
        super(LeNet_internal3, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.batchnorm3 = nn.BatchNorm1d(120)
        self.batchnorm4 = nn.BatchNorm1d(84)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.internal1 = None
        self.internal2 = None
        self.internal3 = None
        self.internal4 = None

    
    def forward(self, x):
        x = F.sigmoid(self.batchnorm1(self.conv1(x)))
        self.internal1 = x.clone()
        x = self.pool(x)
        x = F.sigmoid(self.batchnorm2(self.conv2(x)))
        self.internal2 = x.clone()
        x = self.pool(x)
        x = x.view(-1, 5*5*16)
        x = F.sigmoid(self.batchnorm3(self.fc1(x)))
        self.internal3 = x.clone()
        return self.internal3


class LeNet_internal4(nn.Sequential):
    
    def __init__(self):
        super(LeNet_internal4, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.batchnorm3 = nn.BatchNorm1d(120)
        self.batchnorm4 = nn.BatchNorm1d(84)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.internal1 = None
        self.internal2 = None
        self.internal3 = None
        self.internal4 = None

    
    def forward(self, x):
        x = F.sigmoid(self.batchnorm1(self.conv1(x)))
        self.internal1 = x.clone()
        x = self.pool(x)
        x = F.sigmoid(self.batchnorm2(self.conv2(x)))
        self.internal2 = x.clone()
        x = self.pool(x)
        x = x.view(-1, 5*5*16)
        x = F.sigmoid(self.batchnorm3(self.fc1(x)))
        self.internal3 = x.clone()
        x = F.sigmoid(self.batchnorm4(self.fc2(x)))
        self.internal4 = x.clone()
        return self.internal4

class MPCLeNet(nn.Sequential):
    
    def __init__(self):
        super(MPCLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.batchnorm3 = nn.BatchNorm1d(120)
        self.batchnorm4 = nn.BatchNorm1d(84)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.internal1 = None
        self.internal2 = None
        self.internal3 = None
        self.internal4 = None

    def mpcbatchnorm1(self, x):
        return mpcbatchnorm2d(x, self.batchnorm1.running_mean, self.batchnorm1.running_var, self.batchnorm1.weight.data, self.batchnorm1.bias.data, self.batchnorm1.eps)
    def mpcbatchnorm2(self, x):
        return mpcbatchnorm2d(x, self.batchnorm2.running_mean, self.batchnorm2.running_var, self.batchnorm2.weight.data, self.batchnorm2.bias.data, self.batchnorm2.eps)
    def mpcbatchnorm3(self, x):
        return mpcbatchnorm1d(x, self.batchnorm3.running_mean, self.batchnorm3.running_var, self.batchnorm3.weight.data, self.batchnorm3.bias.data, self.batchnorm3.eps)
    def mpcbatchnorm4(self, x):
        return mpcbatchnorm1d(x, self.batchnorm4.running_mean, self.batchnorm4.running_var, self.batchnorm4.weight.data, self.batchnorm4.bias.data, self.batchnorm4.eps)

    
    def forward(self, x):
        x = self.pool(mpcsigmoid(self.mpcbatchnorm1(self.conv1(x))))
        self.internal1 = x.clone()
        x = self.pool(mpcsigmoid(self.mpcbatchnorm2(self.conv2(x))))
        x = x.view(-1, 5*5*16)
        self.internal2 = x.clone()
        x = mpcsigmoid(self.mpcbatchnorm3(self.fc1(x)))
        self.internal3 = x.clone()
        x = mpcsigmoid(self.mpcbatchnorm4(self.fc2(x)))
        self.internal4 = x.clone()
        x = self.fc3(x)
        return x

class MPCLeNetAUG(nn.Sequential):
    
    def __init__(self, index_internal1_max=None, index_internal1_min=None, index_internal2=None, index_internal3=None, index_internal4=None):
        super(MPCLeNetAUG, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.batchnorm3 = nn.BatchNorm1d(120)
        self.batchnorm4 = nn.BatchNorm1d(84)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.internal1 = None
        self.internal2 = None
        self.internal3 = None
        self.internal4 = None
        self.index_internal1_max = index_internal1_max
        self.index_internal1_min = index_internal1_min
        self.index_internal2 = index_internal2
        self.index_internal3 = index_internal3
        self.index_internal4 = index_internal4

    def mpcbatchnorm1(self, x):
        return mpcbatchnorm2d(x, self.batchnorm1.running_mean, self.batchnorm1.running_var, self.batchnorm1.weight.data, self.batchnorm1.bias.data, self.batchnorm1.eps)
    def mpcbatchnorm1_highAcc(self, x):
        return mpcbatchnorm2d_highAcc(x, self.batchnorm1.running_mean, self.batchnorm1.running_var, self.batchnorm1.weight.data, self.batchnorm1.bias.data, self.batchnorm1.eps)
    def mpcbatchnorm1_lowAcc(self, x):
        return mpcbatchnorm2d_lowAcc(x, self.batchnorm1.running_mean, self.batchnorm1.running_var, self.batchnorm1.weight.data, self.batchnorm1.bias.data, self.batchnorm1.eps)
    def mpcbatchnorm2(self, x):
        return mpcbatchnorm2d(x, self.batchnorm2.running_mean, self.batchnorm2.running_var, self.batchnorm2.weight.data, self.batchnorm2.bias.data, self.batchnorm2.eps)
    def mpcbatchnorm2_highAcc(self, x):
        return mpcbatchnorm2d_highAcc(x, self.batchnorm2.running_mean, self.batchnorm2.running_var, self.batchnorm2.weight.data, self.batchnorm2.bias.data, self.batchnorm2.eps)
    def mpcbatchnorm3(self, x):
        return mpcbatchnorm1d(x, self.batchnorm3.running_mean, self.batchnorm3.running_var, self.batchnorm3.weight.data, self.batchnorm3.bias.data, self.batchnorm3.eps)
    def mpcbatchnorm3_highAcc(self, x):
        return mpcbatchnorm1d_highAcc(x, self.batchnorm3.running_mean, self.batchnorm3.running_var, self.batchnorm3.weight.data, self.batchnorm3.bias.data, self.batchnorm3.eps)
    def mpcbatchnorm4(self, x):
        return mpcbatchnorm1d(x, self.batchnorm4.running_mean, self.batchnorm4.running_var, self.batchnorm4.weight.data, self.batchnorm4.bias.data, self.batchnorm4.eps)
    def mpcbatchnorm4_highAcc(self, x):
        return mpcbatchnorm1d_highAcc(x, self.batchnorm4.running_mean, self.batchnorm4.running_var, self.batchnorm4.weight.data, self.batchnorm4.bias.data, self.batchnorm4.eps)

    
    def forward(self, x):
        if self.index_internal1_min and self.index_internal1_max:
            x1 = self.pool(torch.sigmoid(self.mpcbatchnorm1_highAcc(self.conv1(x.clone()))))
            x2 = self.pool(mpcsigmoid(self.mpcbatchnorm1_lowAcc(self.conv1(x.clone()))))
            x = self.pool(mpcsigmoid(self.mpcbatchnorm1(self.conv1(x))))
            for internal1_index in self.index_internal1_max:
                # print(x[internal1_index], x1[internal1_index])
                x[internal1_index] = x1[internal1_index]
            for internal1_index in self.index_internal1_min:
                x[internal1_index] = x2[internal1_index]
        else:
            x = self.pool(mpcsigmoid(self.mpcbatchnorm1(self.conv1(x))))
        self.internal1 = x.clone()
        x = self.pool(mpcsigmoid(self.mpcbatchnorm2(self.conv2(x))))
        x = x.view(-1, 5*5*16)
        self.internal2 = x.clone()
        x = mpcsigmoid(self.mpcbatchnorm3(self.fc1(x)))
        self.internal3 = x.clone()
        x = mpcsigmoid(self.mpcbatchnorm4(self.fc2(x)))
        self.internal4 = x.clone()
        x = self.fc3(x)
        return x


def preprocess_data(args, context_manager, data_dirname):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    with context_manager:
        trainset = datasets.MNIST(
            './dataset', train=True, download=False, transform=transform
        )
        testset = datasets.MNIST(
            './dataset', train=False, download=False, transform=transform
        )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    return trainloader, testloader

def accuracy(output, target, topk=(1,), ind=0):
    """Computes the precision@k for the specified values of k"""
    global store_wrong_index
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
            if k == 1 and res[-1] < 100:
                index_of_zero = np.where(correct[0].flatten() == 0)[0]
                for p in index_of_zero:
                    store_wrong_index.add(ind * batch_size + p)
        return res


def validate(args, val_loader, model, criterion, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    global store_wrong_index
    store_wrong_index = set()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if isinstance(model, crypten.nn.Module) and not crypten.is_encrypted_tensor(
                input
            ):
                input = encrypt_data_tensor_with_src(input)
            output = model(input)
            if crypten.is_encrypted_tensor(output):
                output = output.get_plain_text()
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5), ind=i)
            losses.add(loss.item(), input.size(0))
            top1.add(prec1[0], input.size(0))
            top5.add(prec5[0], input.size(0))

            current_batch_time = time.time() - end
            batch_time.add(current_batch_time)
            end = time.time()

            if (i + 1) % print_freq == 0:
                logging.info(
                    "\nTest: [{}/{}]\t"
                    "Time {:.3f} ({:.3f})\t"
                    "Loss {:.4f} ({:.4f})\t"
                    "Prec@1 {:.3f} ({:.3f})   \t"
                    "Prec@5 {:.3f} ({:.3f})".format(
                        i + 1,
                        len(val_loader),
                        current_batch_time,
                        batch_time.value(),
                        loss.item(),
                        losses.value(),
                        prec1[0],
                        top1.value(),
                        prec5[0],
                        top5.value(),
                    )
                )

        logging.info(
            " * Prec@1 {:.3f} Prec@5 {:.3f}".format(top1.value(), top5.value())
        )
    print(store_wrong_index)
    return top1.value()

@mpc.run_multiprocess(world_size=2)
def verify_crypt(inputs, model):
    inputCrypt, _, _, _, _ = inputs[0]
    input_size = inputCrypt.size()
    dummy_input = torch.empty(input_size)
    private_model = crypten.nn.from_pytorch(model, dummy_input).encrypt(src=0)
    RealAECount = 0
    AEIndex = []
    for i in range(len(inputs)):
        inputCrypt, precOrig, precMPC, target, ind = inputs[i]
        private_input = crypten.cryptensor(inputCrypt, src=1)
        OutputCrypt = private_model(private_input)
        OutputCrypt = OutputCrypt.get_plain_text()
        precCryp, _ = accuracy(OutputCrypt, target, topk=(1, 5), ind=ind)
        if precCryp < precOrig:
            RealAECount += 1
            AEIndex.append(i)
    # rank = comm.get().get_rank()
    # if rank == 1:
    return AEIndex

def saving(image, index):
    img_path = './results/AE' + str(index) + '.png'
    image = image * 0.3081 + 0.1307
    print(image.max(), image.min())
    save_image(image, img_path)

def generateAE(args, val_loader, modelMPC, modelOrig, modelCryp, criterion, print_freq=10):
    modelMPC.eval()
    global store_wrong_index
    store_wrong_index = set()

    end = time.time()
    AEcount = 0
    RealAEcount = 0
    store_AEs = []
    pbar = tqdm.tqdm(total=len(val_loader))
    for i, (input, target) in enumerate(val_loader):
        input.requires_grad = True
        optimizer = optim.SGD([input], lr=args.lr, momentum=0.9)
        for k_iter in range(args.iter):
            outputMPC = modelMPC(input)
            outputOrig = modelOrig(input)
            optimizer.zero_grad()
            loss = torch.abs(modelOrig.internal1 - modelMPC.internal1).sum()
            loss.backward(retain_graph=True)
            optimizer.step()
            input.data.clamp_(min=-0.1307/0.3081, max=(1-0.1307)/0.3081)
            precMPC, _ = accuracy(outputMPC, target, topk=(1, 5), ind=i)
            precOrig, _ = accuracy(outputOrig, target, topk=(1, 5), ind=i)
            with torch.no_grad():
                input_for_Cryp = input.clone()
                if precMPC < precOrig:
                    AEcount += 1
                    store_AEs.append((input_for_Cryp, precOrig, precMPC, target.clone(), i))
                    break
        del input, optimizer
        pbar.update(1)
        pbar.set_description("Count numbers: Total AEs %d, Real AEs %d" % (AEcount, RealAEcount))

    with open("./analyze/store_AEs.pkl", "wb") as fp:
        pickle.dump(store_AEs, fp)

    AEIndex = verify_crypt(store_AEs, modelCryp)
    print(AEIndex)
    for i in AEIndex[0]:
        image_to_store, _, _, _, ind = store_AEs[i]
        saving(image_to_store, ind)

    print(AEcount, len(AEIndex[0]))

    return RealAEcount

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch_num', type=int, default=25)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--iter', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model_path', type=str, default='./model/MNIST_LeNet_sigmoid_batchnorm.pth')
    parser.add_argument('--check_Crypten', default=False, action="store_true")
    args = parser.parse_args()
    device = "cpu"
    args.device = device
    logging.basicConfig(level = logging.CRITICAL)
    modelMPC = MPCLeNet()
    modelMPC.load_state_dict(torch.load(args.model_path,  map_location=torch.device('cpu')))
    modelMPC.eval()

    X = torch.Tensor([10])
    print(mpcsigmoid(X))
    print(mpcsigmoid_highAcc(X))
    print(torch.sigmoid(X))