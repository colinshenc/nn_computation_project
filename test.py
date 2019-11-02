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
from manipulate import *

def test(model, test_loader, device, loss, perturbed, count_shuffled_unshuffled):
    test_loss = 0
    accuracy = 0
    model.eval()
    total_shuffled_count = 0
    total_unshuffled_count = 0
    total_correct = 0
    with torch.no_grad():
        for batch, label in test_loader:
            #data = data.to(device = device)
            #target = target.to(device = device)

            #model = model.to(device=device, dtype=torch.double)
            batch = batch.to(device=device, dtype=torch.double)
            label = label.to(device=device, dtype=torch.long)

            out = model(batch)
            #print("out{}".format(out.size()))

            #out = out.squeeze(-1).squeeze(-1)
            #print("out{}".format(out.size()))
            pred_class = out.max(1, keepdim=True)[1]#get indices of class
            #print("pred_class size {}".format(pred_class.size()))
            #print("out size {}".format(out.size()))
            #print("label size {}".format(label.size()))
            if count_shuffled_unshuffled:
                new_label = restore_labels(label)
                test_loss += loss(out, new_label).item()
                accuracy += pred_class.eq(new_label.view_as(pred_class)).sum().item()
            #print('accuracy count {}'.format(accuracy))
            #print(len(test_loader.dataset))
            #print(test_loader.batch_size)
            #print("accuracy {}".format(accuracy))
                shuffled_memo_count, unshuffled_memo_count = get_memo_counts(pred_class, label)
                total_shuffled_count += shuffled_memo_count
                total_unshuffled_count += unshuffled_memo_count
                del new_label
            else:
                test_loss += loss(out, label).item()
                accuracy += pred_class.eq(label.view_as(pred_class)).sum().item()
            del label
            del batch
    accuracy_percent = accuracy * 100.0 / len(test_loader.dataset)
    test_loss = test_loss * 1.0 / len(test_loader.dataset)
    # shuffled_accuracy = total_shuffled_count * 100.0 / len(test_loader.dataset)
    # unshuffled_accuracy = total_unshuffled_count * 100.0 / len(test_loader.dataset)

    # loss_title = "val_loss"
    # acc_title = "val accuracy percent"
    #if perturbed:
    #    loss_title = "{}_perturbed".format(loss_title)
    #    acc_title = "{}_perturbed".format(acc_title)
    # if len(args) != 0:
    #     #pass
    #     #print("args{}
    #     #print("b")
    #     sw.add_scalar(loss_title, test_loss, args[0])
    #     sw.add_scalar(acc_title, accuracy, args[0])
        #print("b")
    #print("{}{}".format(loss_title, test_loss))
    #print("{}: {}".format(acc_title, accuracy))
    #if (not perturbed) and count_shuffled_unshuffled:
    return test_loss, accuracy, total_shuffled_count, total_unshuffled_count
    #else:
        #return test_loss, accuracy_percent

# def run_test(model_file_paths, manipulate_weights, shuffle_labels, project_path, exp_names, *args):
#     for model_path in model_file_paths:
#
#         loss_arr = []
#         acc_arr = []
#         range_ = []
#         shuffle_ratios = []
#         if manipulate_weights:
#             range_ = np.arange(-args["t"], args["t"], args["delta"])
#             noise_state_dict = generate_noise_state_dict(model_path)
#
#         if shuffle_labels:
#             shuffle_ratios = np.arange(args["shuffle_low"], args["shuffle_high"], args["granularity"])
#
#         sw = SW(logdir="{}{}{}_perturbed".format(project_path, "/runs/", exp_names[model_file_paths.index(model_path)]))  # log_dir="./logdir/" + exp_name)
#         for perturbation_ratio in range_:
#             for shuffle_ratio in shuffle_ratios:
#                 model = load_weights(model_path, manipulate_weights, perturbation_ratio, device, noise_state_dict)
#             loss, acc = test(model, test_loader, device, loss_func, sw, True)
#             loss_arr.append(loss)
#             acc_arr.append(acc)
#         print("acc {}".format(acc_arr))
#         print("loss {}".format(loss_arr))
#
#         loss_title = "loss after perturbation"
#         acc_title = "accuracy after perturbation"
#         acc_fig = plot_figure(range_, acc_arr, acc_title, exp_names[model_file_paths.index(model_path)])
#         loss_fig = plot_figure(range_, loss_arr, loss_title, exp_names[model_file_paths.index(model_path)])
#         sw.add_figure("figures", [acc_fig, loss_fig])
#         sw.close()

