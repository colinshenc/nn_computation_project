import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter as SW
from utils.utils import *


def train_step(model, batch, label, device, optimizer, loss, *init_func):
    #batch_loss = 0 #torch.tensor([1e-10], dtype=torch.double, device=self.device)
    # if not(type(init_func) == str and len(init_func) > 0):
    #     raise AssertionError("init_func must be of type str and not empty!")


    #model = model.to(device=device, dtype=torch.double)
    batch = batch.to(device=device, dtype=torch.double)
    label = label.to(device=device, dtype=torch.long)
    #print('lab size {}'.format(label.size()))
    model.train()
    #print("batch size {}".format(batch.size()))
    model.zero_grad()
    optimizer.zero_grad()
    output = model(batch)
    #print("output size {}".format(output.size()))

    #output = output.squeeze(-1).squeeze(-1)
    #print("output after{}".format(output.size()))
    #print("label {}".format(label))
    #label_one_hot = convert_to_one_hot(label, num_cls = 10)
    #print("label size {}".format(label.size()))
    loss = loss(output, label)
    #print("loss{}".format(loss))
    loss.backward(retain_graph=False)
    # for param in model.parameters():
    #     print(param.grad.data.sum())

    del output
    del batch
    del label

    optimizer.step()
    torch.cuda.empty_cache()
    # model.zero_grad()
    # optimizer.zero_grad()
    #print("loss{}".format(loss))
    #print(loss)
    return loss


# def train_batch(model, train_loader, epochs, device, model_optimizer, loss_func, init_func):
#         loss = train_step(model, batch, label, device, model_optimizer, loss_func, init_func)
#
#
# def train_shuffle(model, epochs):
#
#
#
# def train(model, epochs):
#     for epoch in range(epochs):
#         total_loss = 0
#         for idx, (batch, label) in enumerate(train_loader):
#             # print("batch {}".format(batch.size()))
#             # print("label {}".format(label.size()))
#             shuffle_batch_label(label, shuffle_ratio)
#             loss = train_step(model, batch, label, device, model_optimizer, loss_func, init_func)
#             # print(loss)
#             # print(loss)
#             total_loss += loss
#
#         print(epoch)
#         total_loss_ = total_loss * 1.0 / len(train_loader.dataset)
#         print("train loss {}".format(total_loss_))
#         sw.add_scalar('train_loss', total_loss_, epoch)
#         test(model, test_loader, device, loss_func, sw, False, epoch)
#         sw.close()
#         print("train loss/epoch {}".format(total_loss))
