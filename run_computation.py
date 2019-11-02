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
import torchvision.models as models
import datetime

import time

from train import train_step
import json
from utils.serialization import write_to_json

from test import test
from manipulate import *

from functools import partial
from utils.utils import *
import matplotlib.pyplot as plt
import logging
import random
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--which_dataset', required=True, choices=['mnist','fashion_mnist','cifar10'])
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--lr', type=float, default=4e-5)
parser.add_argument('--which_scheduler', required=True, choices=['exponential', 'step'])
parser.add_argument('--num_experiment_reps', default=1)
parser.add_argument('--which_conv_init_func', default='kaiming_normal', choices=['kaiming_normal', 'normal', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform'])
parser.add_argument('--which_bn_init_func', default='normal', choices=['normal', 'uniform', 'constant'])
parser.add_argument('--shuffle_ratio_low', required=True, type=float, default=0.0)
parser.add_argument('--shuffle_ratio_high', required=True, type=float, default=1.1)
parser.add_argument('--shuffle_step', type=float, default=0.1)
parser.add_argument('--count_shuffled_unshuffled', type=bool, default=True)
parser.add_argument('--number_perturbation_tensor', type=int, default=15)
parser.add_argument('--delete_model_after_experiment', type=bool, default=True)
parser.add_argument('--max_perturbation', required=True, type=int)
parser.add_argument('--perturbation_step', required=True, type=float)

args = parser.parse_args()




















###############model parameters##################
datasets_= {'mnist':datasets.MNIST,
            'fashion_mnist':datasets.FashionMNIST,
            'cifar10':datasets.CIFAR10
            }

transforms={'mnist':transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))]),
            'fashion_mnist':transforms.Compose([
                               #transforms.Resize(224),
                               #transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.5,), std=(0.5,))]),
            'cifar10':transforms.Compose([
                transforms.CenterCrop(28),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            }
#C=3 for cifar10
#batch_size = {'mnist':[275], 'fashion_mnist':[475], 'cifar10':[100]}
schedulers={'step':torch.optim.lr_scheduler.StepLR,
            'exponential':torch.optim.lr_scheduler.ExponentialLR,
            }
#which_scheduler='exponential'
shuffle_or_not = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__)
print("training on {}".format(device))
# set up cudnn benchmark for better speed
torch.backends.cudnn.benchmark = True
optimizer_name = "Adam"
#lr = [4e-5]
num_out_class = {'mnist':10, 'fashion_mnist':10, 'cifar10' : 10}
#conv_block_in_out = [1, 4, 8, 10]# last element must match first element.
#lin_layer_in_out = [1, 40, 160, 100, 50, 10, 2]  # first is first layer out, last = 10
#conv_filter_size = 3
#repetitions = 1
#epochs = 1
#pool_size = 2
#initialization parameters
uni_a = 0.0
uni_b = 0.8
norm_mean = 0.0
norm_std = 0.5
const = 0.0
#which_dataset='fashion_mnist'

# save_data = False #toggle to choose whether to save data
init_funcs_conv = {'kaiming_normal':partial(eval("torch.nn.init.kaiming_normal_"), mode='fan_out', nonlinearity='relu'),
                  'normal':partial(eval("torch.nn.init.normal_"), mean=norm_mean, std=norm_std),
                  'xavier_uniform':eval("torch.nn.init.xavier_uniform_"),
                  'xavier_normal':eval("torch.nn.init.xavier_normal_"),
                  'kaiming_uniform':eval("torch.nn.init.kaiming_uniform_")
                   }

init_funcs_bn = {'normal':partial(eval("torch.nn.init.normal_"), mean=norm_mean, std = norm_std),
                'uniform':partial(eval("torch.nn.init.uniform_"), a=uni_a, b=uni_b),
                'constant':partial(eval("torch.nn.init.constant_"), val=const)
                 }

# uniform_init_func = partial(eval("torch.nn.init.uniform_"), a=uni_a, b=uni_b)
# normal_init_func = partial(eval("torch.nn.init.normal_"), mean=norm_mean, std = norm_std)
# constant_init_func = partial(eval("torch.nn.init.constant_"),val=const)
#kaiming_init_func = partial(eval("torch.nn.init.kaiming_normal_"), mode='fan_out', nonlinearity='relu')
                  #choose initialization function here.
#change_params = True #whether we re-touch trained weights
#choose batch norm initialization here
#init_func_bn = normal_init_func

train = True
#t = 5
#delta = 2 #granularity for ploting
#shuffle_step = 0.1
manipulate_weights = True
shuffle_label = True
#shuffle_high = 0.4
combine_dataset = True
#number_perturbation_tensor = 1
#count_shuffled_unshuffled = True
###################################################

#logging

# logging.basicConfig(filename='code.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.DEBUG)
# logging.debug('log starts here...')




###################################################



project_path = "/home/colin/Downloads/nn_computation_project"
exp_name = "{}_exp_{}".format(args.which_dataset, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
model_file_paths = []
exp_names = []
#root path for test, train set, set new ones for new datasets
dataset_root = "/home/colin/Downloads/nn_computation_project/{}_train_data".format(args.which_dataset)
#test_set_root = "/home/colin/Downloads/nn_computation_project/test_data"

json_weight_file_name = "weight_{}".format(exp_name)
json_bias_file_name = "bias_{}".format(exp_name)
json_weight_file_path = "{}/output_data/{}".format(project_path, json_weight_file_name)
json_bias_file_path = "{}/output_data/{}".format(project_path, json_bias_file_name)

# train_set = datasets.MNIST(root=train_set_root, train=True, download=True, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ]))
# test_set = datasets.MNIST(root=test_set_root, train=train_on_test_set, download=True, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ]))


#print("{} train set size {}".format(dataset_name, len(train_set)))
#print("{} test set size {}".format(dataset_name, len(test_set)))

#
# train_loader = DataLoader(
#     dataset = train_set,
#     batch_size = batch_size,
#     shuffle = shuffle_or_not,
#     num_workers = 0)
#
# #test_loader = DataLoader(
#     dataset = test_set,
#     batch_size = batch_size,
#     shuffle = shuffle_or_not,
#     num_workers = 0)

shuffle_dict = dict(zip(range(num_out_class[args.which_dataset]),random.sample(range(num_out_class[args.which_dataset]), num_out_class[args.which_dataset])))
#model = NET(#conv_block_in_out,#last element must match first element.
            # lin_layer_in_out,
            # max_pool_size=pool_size,#first is first layer out
            # conv_filter_size = 3,
            # num_class = num_out_class
 #           ).to(device)
#
#model_optimizer =Adam(params=model.parameters(), lr=lr)
#print(model)

loss_func = nn.CrossEntropyLoss()
torch.autograd.set_detect_anomaly(True)
#SW = SW(logdir="{}{}{}".format(project_path,"/runs/", exp_name))  # log_dir="./logdir/" + exp_name)
# print("save data? {}".format(save_data))

#try:
#training loop
if train:
    #sw = SW(logdir="{}{}_perturbed_{}".format(project_path, "/runs/", exp_name)) # log_dir="./logdir/" + exp_name)

    for repetition in tqdm(range(args.num_experiment_reps)):
        model = NET(  # conv_block_in_out,#last element must match first element.
            # lin_layer_in_out,
            # max_pool_size=pool_size,#first is first layer out
            # conv_filter_size = 3,
            # num_class = num_out_class
        ).to(device)
#        for lr_ in lr:
#           for batch_size_ in [args.batch_size]:

        # train_loader = DataLoader(
        #     dataset=train_set,
        #     batch_size=batch_size_,
        #     shuffle=shuffle_or_not,
        #     pin_memory=True,#important
        #     num_workers=0)
        #
        # test_loader = DataLoader(
        #     dataset=test_set,
        #     batch_size=batch_size_,
        #     shuffle=shuffle_or_not,
        #     num_workers=0)

        exp_name_ = "{}_lr_{}_epoch_{}_batch_size_{}".format(exp_name,args.lr, args.epochs, args.batch_size)
        #save_path_model = "{}/models/{}.pt".format(project_path,exp_name)
        #sw = SW(logdir="{}{}{}".format(project_path, "/runs/", exp_name_))  # log_dir="./logdir/" + exp_name)
        exp_name_ = "{}_num_trainable_params_{}".format(exp_name_, sum(
            p.numel() for p in model.parameters() if p.requires_grad))
        print("model param count {}".format(sum(p.numel() for p in model.parameters())))
        print("of which trainable {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        json_weight_path_with_rep = "{}_rep_{}_lr_{}_b_size_{}.json".format(json_weight_file_path, str(repetition),args.lr,args.batch_size)
        json_bias_path_with_rep = "{}_rep_{}_lr_{}_b_size_{}.json".format(json_bias_file_path, str(repetition),args.lr,args.batch_size)
        #initialize the weights
        #model.init_weights(init_func=init_func, init_func_bn=init_func_bn)

        #model_optimizer = Adam(params=model.parameters(), lr=lr_)
        exps_dict = {}
        if shuffle_label:
            shuffle_range = np.arange(args.shuffle_ratio_low, args.shuffle_ratio_high, args.shuffle_step)
            acc_y = []
            loss_y = []
            exp_name_ = "{}_shuffle_labels".format(exp_name_)
            save_path_model = "{}/models/{}.pt".format(project_path, exp_name_)
            #sw = SW(logdir="{}{}{}".format(project_path, "/runs/", exp_name_))  # log_dir="./logdir/" + exp_name)
            accu_shuffled = []
            accu_unshuffled = []
            tot_accu = []
            for shuffle_ratio in shuffle_range:
                train_set = datasets_[args.which_dataset](root=dataset_root, train=True, download=True,
                                            transform=transforms[args.which_dataset]
                                                      )

                test_set = datasets_[args.which_dataset](root=dataset_root, train=False, download=True,
                                            transform=transforms[args.which_dataset]
                                                    )

                print("{} train set size {}".format(args.which_dataset, len(train_set)))
                print("{} test set size {}".format(args.which_dataset, len(test_set)))
                print("shuffle ratio {}".format(shuffle_ratio))

                shuffle_set_labels(train_set, shuffle_ratio, shuffle_dict, args.count_shuffled_unshuffled)

                train_loader, test_loader = prepare_loader(combine_dataset, train_set, test_set, args.batch_size,
                                                     shuffle_or_not)
                #print(len(train_loader))
                # print('type{}'.format(train_set.targets))
                #del model
                ###important: when combine_dataset is true, train data contains test data(for particular reasons...)
                if combine_dataset:
                    total_loss = 0.0
                    model = NET().to(device=device, dtype=torch.double)
                    #print("model param count {}".format(sum(p.numel() for p in model.parameters())))
                    #print("of which trainable {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
                    model.init_weights(init_func_conv=init_funcs_conv[args.which_conv_init_func], init_func_bn=init_funcs_bn[args.which_bn_init_func])
                    model_optimizer = Adam(params=model.parameters(), lr=args.lr)
                    #exp_name_ = "{}_num_trainable_params_{}".format(exp_name_, sum(p.numel() for p in model.parameters() if p.requires_grad))
                    scheduler = schedulers[args.which_scheduler](model_optimizer, 0.99)
                    print(exp_name_)
                    for epoch in range(args.epochs):
                        #print(exp_name_)
                        #print(torch.cuda.get_device_name(torch.cuda.current_device()))
                        for idx, (batch, label) in enumerate(train_loader, 0):
                            # print("batch {}".format(batch.size()))
                            # print("label grad{}".format(label.requires_grad))
                            # print("batch{}".format(idx))
                            # new_label = label.clone().detach()
                            # new_label = new_label.to(device)
                            # print('label size {}'.format(label.size()))
                            if args.count_shuffled_unshuffled:
                                new_label = restore_labels(label)
                            #new_label, ratio = shuffle_batch_label(label, shuffle_ratio, device)
                            #new_label = new_label.detach()
                                loss = train_step(model, batch, new_label, device, model_optimizer, loss_func)
                                #del new_label
                            else:
                                loss = train_step(model, batch, label, device, model_optimizer, loss_func)
                            #del label
                            #del batch
                            #print("ratio: {}".format(ratio))
                            loss = loss.detach() #important!
                            total_loss += loss.item()
                            #del batch
                            del loss
                            #del label, new_label


                        #print(epoch)
                        print("epoch: {}".format(epoch))
                        # test_loss, accuracy, shuffled_accuracy, unshuffled_accuracy = test(model, test_loader, device, loss_func, [], False,
                        #                            shuffle_ratio)

                        #print("acc: {}".format(accuracy))
                        scheduler.step()

                    test_loss, accuracy, shuffled_accuracy, unshuffled_accuracy = test(model, train_loader,
                                                                                       device, loss_func,
                                                                                       False, args.count_shuffled_unshuffled)
                    if args.count_shuffled_unshuffled:
                        accu_shuffled.append(shuffled_accuracy)
                        accu_unshuffled.append(unshuffled_accuracy)
                    tot_accu.append(accuracy)
                    total_loss_ = total_loss * 1.0 / len(train_loader.dataset)
                    save_path_model = "{}/models/{}.pt".format(project_path, "{}_ratio_{}".format(exp_name_, shuffle_ratio))



                    torch.save(model.state_dict(), save_path_model)
                    exps_dict["{}_ratio_{:0.2f}".format(exp_name_, shuffle_ratio)] = save_path_model
                    #exp_names.append("{}_ratio_{}".format(exp_name_, shuffle_ratio))
                    #model_file_paths.append(save_path_model)
                    #print("shuffle ratio {}".format(shuffle_ratio))
                    print("train loss {}".format(total_loss_))
                    del total_loss_, total_loss
                    del model
                    del model_optimizer
                    del scheduler
                    del train_set
                    del test_set
                    del train_loader
                    del test_loader
            os.mkdir("{}{}_perturbed_{}".format(project_path, '/runs/', exp_name))
            if args.count_shuffled_unshuffled:
                bar_plot(accu_shuffled, accu_unshuffled, tot_accu, shuffle_range,
                             "{}{}_perturbed_{}".format(project_path, '/runs/', exp_name))
                    #sw.add_scalar('train_loss', total_loss_, shuffle_ratio)
                    #test_loss, accuracy = test(model, test_loader, device, loss_func, [], False, shuffle_ratio)
                    #print("acc: {}".format(accuracy))
                    #acc_y.append(accuracy)
                    #loss_y.append(test_loss)

                #acc_fig = plot_figure(X, acc_y, "acc", exp_name_)
                #loss_fig = plot_figure(X, loss_y, "loss", exp_name_)
                #sw.add_figure("figures", [acc_fig, loss_fig])
            #sw.close()

                #print("train loss/epoch {}".format(total_loss))
                #print(model.state_dict().keys())
                #print(model.state_dict()['conv1.bias'])

                #
                # if save_data:
                #     weight_dict = model.extract_weight(repetition)
                #     bias_dict = model.extract_bias(repetition)
                #     #print(json_weight_path_with_rep)
                #     write_to_json(weight_dict, json_filename=json_weight_path_with_rep)
                #     write_to_json(bias_dict, json_filename=json_bias_path_with_rep)



#print("000")
if manipulate_weights:
    #model = NET()
    max_loss_dict = {}
    avg_loss_dict = {}
    max_acc_dict = {}
    avg_acc_dict = {}
    # sw = SW(logdir="{}{}_perturbed_{}".format(project_path, "/runs/", exp_name)) # log_dir="./logdir/" + exp_name)
    range_ = np.arange(0, args.max_perturbation, args.perturbation_step)
    test_set = datasets_[args.which_dataset](root=dataset_root, train=False, download=True, transform=transforms[args.which_dataset])
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=shuffle_or_not,
        drop_last=False,
        num_workers=0)

    #print(111111)

    for k, v in exps_dict.items():
        print(k)
        #noise_state_dict = []
        #range_ = np.arange(-t, t, delta)
        #noise_state_dict = generate_noise_state_dict(v)
        #sw = SW(logdir="{}{}{}_perturbed".format(project_path, "/runs/", exp_names[model_file_paths.index(model_path)]))  # log_dir="./logdir/" + exp_name)
        max_temp_loss = []
        max_temp_acc = []
        avg_temp_loss = []
        avg_temp_acc = []

        for i in range_:
            avg_loss = 0
            datapoint_loss = []
            datapoint_acc = []
            for j in range(args.number_perturbation_tensor):
                noise_state_dict = generate_noise_state_dict(list(exps_dict.values())[0], gen_func=torch.randn_like)
                model = load_weights(v, manipulate_weights, i, device, noise_state_dict)
                loss, acc, _, _ = test(model, test_loader, device, loss_func, True, args.count_shuffled_unshuffled)
                datapoint_loss.append(loss)
                datapoint_acc.append(acc)

            avg_temp_loss.append(sum(datapoint_loss) / len(datapoint_loss))
            avg_temp_acc.append(sum(datapoint_acc) / len(datapoint_acc))
            max_temp_loss.append(max(datapoint_loss))
            max_temp_acc.append(max(datapoint_acc))

        avg_loss_dict[k] = avg_temp_loss
        avg_acc_dict[k] = avg_temp_acc
        max_acc_dict[k] = max_temp_acc
        max_loss_dict[k] = max_temp_loss
    if args.delete_model_after_experiment:
        for _, v in exps_dict.items():
            os.remove(v)

        #print("acc {}".format(acc_arr))
        #print("loss {}".format(loss_arr))

    avg_loss_title = "avg. loss after perturbation"
    avg_acc_title = "avg. accuracy after perturbation"
    max_loss_title = 'max. loss after perturbation'
    max_acc_title = 'max. accuracy after perturbation'
    plot_figure(range_, avg_loss_dict, avg_loss_title, exp_name, None, "{}{}_perturbed_{}".format(project_path, "/runs/", exp_name))
    plot_figure(range_, avg_acc_dict, avg_acc_title, exp_name, None, "{}{}_perturbed_{}".format(project_path, "/runs/", exp_name))
    plot_figure(range_, max_loss_dict, max_loss_title, exp_name, None, "{}{}_perturbed_{}".format(project_path, "/runs/", exp_name))
    plot_figure(range_, max_acc_dict, max_acc_title, exp_name, None, "{}{}_perturbed_{}".format(project_path, "/runs/", exp_name))

    #sw.add_figure("figures", [acc_fig, loss_fig])
    # sw.close()
        #add_figure(range_, loss_arr, sw, loss_title)
#except Exception as msg:
#    email_msg(exp_name_, msg, datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
