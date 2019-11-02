import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import smtplib
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np

def remove_nesting(list, output):
    for _ in list:
        if type(_) == torch.nn.modules.container.ModuleList:
            remove_nesting(_, output)
        else:
            output.append(_)

def convert_to_one_hot(labels, num_cls):
    one_hot = torch.eye(num_cls)[labels]
    return one_hot


def plot_figure(x, y, title, exp_name, sw, save_path):
    #plt.style.use('seaborn-dark-palette')

    fig = plt.figure(1, figsize=(10, 9))
    #plt.axis([-1.0, 1.0, -3, 25])
    colors = ['dimgray', 'lightcoral', 'maroon', 'red', 'sienna', 'darkorange', 'darkgoldenrod',
              'olive', 'lawngreen', 'darkgreen', 'turquoise', 'aqua', 'dodgerblue', 'royalblue',
              'blueviolet', 'fuchsia', 'crimson']
    markers = ['+', 'x', 'p', 's', '2', 'o', 'P', 'd', 'h', '*']
    for idx, (k, v) in enumerate(sorted(y.items())):
    #fig = plt.figure()
        plt.plot(x, v, colors[idx % len(colors)], marker=markers[idx % len(markers)], label=k, linewidth=1.5)
    plt.xlabel("t")
    plt.ylabel(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0))
    plt.savefig("{}/{}.pdf".format(save_path, title), format='pdf', bbox_inches='tight')

    #sw.add_figure(title, plt)
    #plt.savefig("{}.png".format(title))
    #print("aa")
    #sw.add_figure("figures", [fig])

    fig.clf()



def email_msg(exp_name, msg, time):
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login('jamescolin10@gmail.com','')
    server.sendmail(
        'jamescolin10@gmail.com',
        'jamescolin10@gmail.com',
        "Subject: {}\ntime:{}\n\n{}".format("code error", exp_name, msg)
    )
    server.quit()



def prepare_loader(combine_yes, train_set, test_set, batch_size_, shuffle_or_not):
    # test_loader = DataLoader(
    #     dataset=test_set,
    #     batch_size=batch_size_,
    #     shuffle=shuffle_or_not,
    #     pin_memory=True,
    #     drop_last=False,
    #     num_workers=0)
    #print(len(ConcatDataset([train_set, test_set])))
    if combine_yes:
        #dataset = ConcatDataset([train_set, test_set])
        return DataLoader(
        dataset=ConcatDataset([train_set, test_set]),
        batch_size=batch_size_,
        shuffle=shuffle_or_not,
        pin_memory=True,  # important
        drop_last=False,
        num_workers=0), DataLoader(
        dataset=test_set,
        batch_size=batch_size_,
        shuffle=shuffle_or_not,
        pin_memory=True,
        drop_last=False,
        num_workers=4)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size_,
        shuffle=shuffle_or_not,
        pin_memory=True,
        drop_last=False,
        num_workers=4)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size_,
        shuffle=shuffle_or_not,
        pin_memory=True,  # important
        drop_last=False,
        num_workers=0)

    return train_loader, test_loader

def get_lr(optim):
    for _ in optim.param_groups:
        return _['lr']

def bar_plot(shuffled_arr, unshuffled_arr, accu_arr, shuffle_range, save_path):
    shuffled_arr = np.array(shuffled_arr)
    unshuffled_arr = np.array(unshuffled_arr)
    fig = plt.figure(1, figsize=(10, 9))
    idxs = [_ for _, r in enumerate(shuffle_range)]

    bar = plt.bar(idxs, unshuffled_arr, label='correct unshuffled/untouched labels', color='crimson')
    for _ in bar:
        plt.text(_.get_x()+(0.75*_.get_width())/2, _.get_y()+_.get_height()/2, str(unshuffled_arr[bar.index(_)]), fontsize=9)

    bar = plt.bar(idxs, shuffled_arr, label='correct shuffled labels', color='olive', bottom=unshuffled_arr)
    for _ in bar:
        plt.text(_.get_x(),#+(0.75*_.get_width())/2,
                 _.get_y()+_.get_height()/2, str(shuffled_arr[bar.index(_)]), fontsize=9)
        plt.text(_.get_x(),#+_.get_width()/2,
                 _.get_y()+0.7+_.get_height(), 'tot:{}'.format(accu_arr[bar.index(_)]), fontsize=11)

    plt.xticks(idxs, ['{:0.2f}'.format(_) for _ in shuffle_range])
    plt.xlabel('Shuffle range')
    plt.ylabel('num. labels')
    plt.legend()
    plt.title('Shuffled/unshuffled labels')

    plt.savefig("{}/{}.pdf".format(save_path, 'shuffled_unshuffled_bar_plot'), format='pdf')#, bbox_inches='tight')

    fig.clf()
