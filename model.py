import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
#from torchsummary import summary
from utils.utils import remove_nesting



class NET(nn.Module):

    def __init__(self,
                 # conv_block_in_out,#last element must match first element.
                 # lin_layer_in_out, #first is first layer out
                 # #num_block,
                 # #num_lin_layer,
                 # max_pool_size,
                 # conv_filter_size = 3,
                 # num_class = 10
                 ):

        super().__init__()
        #self.in_channels = in_channels,
        #self.num_block = num_block
        #self.scale_factor = scale_factor
        #self.conv_block_in_out = conv_block_in_out
        #self.lin_layer_in_out
        #self.num_lin_layer = num_lin_layer
        #self.num_nodes = num_nodes
        #self.out_layer = out_layer
        #self.num_out_class = num_out_class
        #self.activation = activation
        #model = nn.ModuleList()
        #self.conv_block = [self.conv_layer]

        #TODO: add a exception for element match for conv_block and lin_layer
#         self.model = nn.ModuleList(
#                      [
#                      nn.ModuleList(
#                      [nn.Conv2d(in_channels, out_channels, conv_filter_size, bias=True),
#                       nn.ReLU(inplace = True),
#                       nn.BatchNorm2d(out_channels),
#                       nn.MaxPool2d(max_pool_size)
#                       ]
#                      )
#                      for in_channels, out_channels in zip(conv_block_in_out[:-1], conv_block_in_out[1:])
#                      ]
# )
#
#         self.model.append(
#             nn.ModuleList(
#                 [nn.ModuleList(
#                     [nn.Linear(in_channels, out_channels, bias=True),
#                         nn.ReLU(inplace=True),
#                      nn.BatchNorm2d(num_class),
#                  ])
#
#                 for in_channels, out_channels in zip(lin_layer_in_out[:-1], lin_layer_in_out[1:])
#                  ])
#         )
#         self.model.append(nn.Linear(lin_layer_in_out[-1], 1,bias=True))
#
#         #remove_nesting(self.model)
#         self.model1 = nn.ModuleList()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=(3, 3), stride=(1, 1))
        self.activ = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(12, 70, kernel_size=(3, 3), stride=(1, 1))
        #self.activ = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(70, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv3 = nn.Conv2d(70, 200, kernel_size=(3, 3), stride=(1, 1))
        #self.activ = nn.ReLU()

        self.bn3 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4 = nn.Conv2d(200, 300, kernel_size=(3, 3), stride=(1, 1))
        #self.mpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.bn4 = nn.BatchNorm2d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        #self.conv5 = nn.Conv2d(150, 500)

        self.lin1 = nn.Linear(in_features=300, out_features=400, bias=True)
        #self.activ = nn.ReLU()bn
        #self.bn4 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.lin2 = nn.Linear(in_features=400, out_features=300, bias=True)
        #self.activ = nn.ReLU()

        #self.bn5 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.lin3 = nn.Linear(in_features=300, out_features=100, bias=True)
        #self.activ = nn.ReLU()
        #self.bn6 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.lin4 = nn.Linear(in_features=100, out_features=50, bias=True)
        #self.activ = nn.ReLU()
        #self.bn7 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.lin5 = nn.Linear(in_features=50, out_features=10, bias=True)
        # self.activ8 = nn.ReLU()
        # self.bn8 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.lin5 = nn.Linear(in_features=10, out_features=2, bias=True)
        # self.activ9 = nn.ReLU()
        # self.bn9 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.lin6 = nn.Linear(in_features=2, out_features=1, bias=True)
        #remove_nesting(self.model, self.model1)
        #del self.model

        self.dropout=nn.Dropout(p=0.2)
        #print(self.model1)
        #self.num_pass = 0

    def forward(self, h):
        #for ix, layer in enumerate(self.model1):

            #h = layer(h)
            #print(h.size())
        #self.num_pass += 1
        #print(h.size())
        #print("before{}".format(h.size()))

        h = self.conv1(h)
        #print(h.size())

        h = self.activ(h)
        #print(h.size())

        h = self.bn1(h)
        #print("after conv1{}".format(h.size()))

        h = self.mpool1(h)
        #print("after mpool1{}".format(h.size()))

        h = self.conv2(h)
        #print(h.size())

        h = self.activ(h)
        #print(h.size())

        h = self.bn2(h)
        #print("after conv2{}".format(h.size()))

        h = self.mpool2(h)
        #print("after mpool2{}".format(h.size()))

        h = self.conv3(h)
        #print(h.size())

        h = self.activ(h)
        #print(h.size())

        h = self.bn3(h)
        #print("after conv3{}".format(h.size()))
        h = self.conv4(h)
        #print(h.size())
        h = self.activ(h)
        #print("after conv4{}".format(h.size()))

        h = h.squeeze().squeeze()
        #print("before lin1{}".format(h.size()))
        h = self.lin1(h)
        #print(h.size())

        h = self.activ(h)
        #print(h.size())
        #h=self.dropout(h)
        #h = self.bn4(h)
        #print(h.size())

        h = self.lin2(h)
        #print(h.size())

        h = self.activ(h)
        #print(h.size())
        #h=self.dropout(h)
        #h = self.bn5(h)
        #print(h.size())

        h = self.lin3(h)
        #print(h.size())

        h = self.activ(h)
        #print(h.size())
        #h=self.dropout(h)
        #h = self.bn6(h)
        #print(h.size())

        h = self.lin4(h)
        #print(h.size())

        h = self.activ(h)
        #print(h.size())
        #h=self.dropout(h)
        #h = self.bn7(h)
        #print(h.size())

        h = self.lin5(h)
        #print(h.size())
        #print("\n\n\n")

        return h

    # def extract_weight(self, repitition):
    #     record = dict()
    #     for idx, layer in enumerate(self.state_dict()):
    #         try:
    #             layer_name = "weights after repitition_{}__layer_{}".format(repitition, idx)
    #             #print("bias size gpu {}".format(layer.weight.data.size()))
    #             #print("weight size cpu {}".format(layer.weight.data.cpu().size()))
    #             _ = layer.weight.data.cpu().numpy().tolist()
    #         except AttributeError:
    #             print("jumping over {}".format(layer_name))
    #             continue
    #         record[layer_name] = _
    #
    #     return record
    #
    # def extract_bias(self, repitition):
    #     record = dict()
    #     for idx, layer in enumerate(self.model1):
    #         try:
    #             layer_name = "biases after repitition_{}__layer_{}".format(repitition, idx)
    #             _ = layer.bias.data.cpu().numpy().tolist()
    #         except AttributeError:
    #             print("jumping over {}".format(layer_name))
    #             continue
    #             record[layer_name] = _
    #
    #     return record

    def init_weights(self, init_func_conv, init_func_bn):
        #print("initializing layer weight with")
        for layer in self.state_dict().keys():
            if (layer.find("conv") != -1 and layer.find("weight") != -1):
#                    or (layer.find("lin") != -1 and layer.find("weight") != -1):

                #print(layer)
                init_func_conv(self.state_dict()[layer])
                #print("layer {}".format(layer))
            elif (layer.find("bn") != -1 and layer.find("weight") != -1) \
                or (layer.find("lin") != -1 and layer.find("weight") != -1):
                #print(layer)
                #print(self.state_dict()[layer].size())
                init_func_bn(self.state_dict()[layer])






def load_vgg(models):
    return models.vgg16_bn()

def load_alexnet(models):
    return models.alexnet()

def load_resnet18(models):
    return models.resnet18()

