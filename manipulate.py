import torch
import torch.nn as nn
from tqdm import tqdm

import torch.nn.functional as F
import torch.optim as optim
import torchvision

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Adam
from model import NET
from tensorboardX import SummaryWriter as SW
import os
import datetime

import time

from train import train_step
import json
from utils.serialization import write_to_json
from collections.abc import Iterable


# def manipulate_lin(layer, i):
#     #print("before {}".format(layer[0]))
#     eps = torch.randn_like(layer) * i
#     layer += eps
#     #print("add {}".format(eps[0]))
#     #print("after {}".format(layer[0]))
#
#
# def manipulate_conv(layer, i):
#     layer += torch.randn_like(layer) * (i * 0.5)


def load_weights(model_path, manipulate_weights, i, device, noise_state_dict):

    model = NET()
    #print(model_path)
    model.to(device)

    state_dict = torch.load(model_path)
    if manipulate_weights:
        for layer in state_dict.keys():
            if layer.find("lin") != -1:
                state_dict[layer] += noise_state_dict[layer] * i

            if layer.find("conv") != -1:
                state_dict[layer] += noise_state_dict[layer] * i
                #manipulate_conv(state_dict[layer], i)
    model.load_state_dict(state_dict)
    return model.to(device=device, dtype=torch.double)


def evaluate_perturbation(i, *args):

    return load_weights(args, i)


def generate_noise_state_dict(model_path,gen_func):
    weights_list = []
    noise_state_dict = torch.load(model_path)
    for layer in noise_state_dict.keys():
        if layer.find("running") == -1:
            temp = gen_func(noise_state_dict[layer], dtype=torch.double)
            #norm = torch.norm(temp.flatten())
            #print('norm {} tensor {}'.format(norm,temp))

            #noise_state_dict[layer] = temp / norm ###normalize in here###
            temp = temp.flatten().tolist()
            if isinstance(temp, Iterable):
                weights_list.extend(temp)
            else:
                weights_list.extend([temp])

    #print(len(weights_list))
    #print('numpy norm {}'.format(np.linalg.norm(weights_list, ord=2)))
    norm = torch.norm(torch.Tensor(weights_list), p='fro')
    #print(norm)
    #a = torch.norm(noise_state_dict, p=2,dim=1)
    for layer in noise_state_dict.keys():
        if layer.find('running') == -1:
            noise_state_dict[layer] = noise_state_dict[layer] / norm

    return noise_state_dict


def sample_label(l):
    list_ = list(range(10))
    list_.remove(l)
    return torch.Tensor([np.random.choice(list_)])


def shuffle_batch_label(label, shuffle_ratio, device):

    rands = torch.rand_like(label, dtype=torch.double).numpy()
    #print(torch.cuda.is_available())
    label = label.numpy()
    #count = 0
    new_label = np.copy(label)
    #label = label.numpy()
    #print(label.device)
    if shuffle_ratio == 0:
        return torch.from_numpy(new_label), 1.0 - (np.count_nonzero(label == new_label)/len(rands))
    for idx, _ in enumerate(rands):

        if _ <= shuffle_ratio:
            #print(_)
            #print("label before {}".format(label[idx]))
            new_label[idx] = sample_label(label[idx])

            #print("label after {}\n".format(label[idx]))
    #print(label.device)
    return torch.from_numpy(new_label), 1.0 - (np.count_nonzero(label == new_label)/len(rands))


def shuffle_set_labels(dataset, ratio, shuffle_dict, count_shuffled_unshuffled):
    if ratio == 0:
        print('zero shuffle rate!')
        return

    if isinstance(dataset.targets, list):
        rands = torch.rand_like(torch.Tensor(dataset.targets), dtype=torch.double)
        for idx, label in enumerate(dataset.targets):
            if rands[idx] <= ratio:
                dataset.targets[idx] = shuffle_dict[dataset.targets[idx]]
                if count_shuffled_unshuffled:
                    dataset.targets[idx] = dataset.targets[idx] + 10

            #del label
        del rands
    else:
        rands = torch.rand_like(dataset.targets, dtype=torch.double)
        for idx, label in enumerate(dataset.targets):
            if rands[idx] <= ratio:
                dataset.targets[idx] = torch.Tensor([shuffle_dict[dataset.targets[idx].item()]])
                if count_shuffled_unshuffled:
                    dataset.targets[idx] = dataset.targets[idx] + 10
            #del label
        del rands

def restore_labels(label):
    new_label = torch.zeros_like(label)
    for idx_, _ in enumerate(new_label):
        if label[idx_] > 9:
            #print(label[idx_])
            new_label[idx_] = label[idx_] - 10
            #print(new_label[idx_])
        else:
            new_label[idx_] = label[idx_]

    return new_label


def get_memo_counts(pred_class, label):
    shuffled_memo_count = 0
    unshuffled_memo_count = 0
    for idx, _ in enumerate(label):
        if label[idx] > 9:
            if (label[idx] - 10) == pred_class[idx]:
                shuffled_memo_count = shuffled_memo_count + 1
        else:
            if label[idx] == pred_class[idx]:
                unshuffled_memo_count = unshuffled_memo_count + 1
    return shuffled_memo_count, unshuffled_memo_count
