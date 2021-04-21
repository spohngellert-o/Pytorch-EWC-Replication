import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import nn, optim
from tqdm import tqdm
import pickle
from collections import OrderedDict
from utils import *

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1200),
            nn.ReLU(),
            nn.Linear(1200, 1200),
            nn.ReLU(),
            nn.Linear(1200, 10)
        )

    def forward(self, x):
        return self.model(x.float())


class DNet(nn.Module):

    def __init__(self):
        super(DNet, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(784, 1200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1200, 1200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1200, 10)
        )

    def forward(self, x):
        return self.model(x.float())



def train(perts, task):
    net = Net() if task == 'EWC' else DNet()
    net.cuda()
    pnets = []
    fishers = []
    train_loaders = []
    test_loaders = []
    val_loaders = []
    test_accs = []
    for i, pert in enumerate(perts):
        val_accs = []
        prev_state_dicts = []
        print("Starting task {}".format(i + 1))
        cur_trset = datasets.MNIST('./files', train=True, transform=transforms.Compose([
            pert,
            transforms.ToTensor()
        ]))
        cur_tset = datasets.MNIST('./files', train=False, transform=transforms.Compose([
            pert,
            transforms.ToTensor()
        ]))
        cur_trset, cur_valset = torch.utils.data.random_split(
            cur_trset, (55000, 5000), generator=torch.Generator().manual_seed(123)
        )
        train_loaders.append(torch.utils.data.DataLoader(
            cur_trset,
            batch_size=10,
            num_workers=2,
            drop_last=False,
            shuffle=True))

        test_loaders.append(torch.utils.data.DataLoader(
            cur_tset,
            batch_size=50,
            num_workers=2,
            drop_last=False))

        val_loaders.append(torch.utils.data.DataLoader(
            cur_valset,
            batch_size=50,
            num_workers=2,
            drop_last=False))
        if task == 'EWC':
            criterion = EWC(fishers, pnets, 0.005)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(net.parameters(), lr=10 ** -3)
        for epoch in range(100):  # loop over the dataset multiple times
            print("Starting epoch {}".format(epoch))
            prev_state_dicts.append(OrderedDict({l: v.cpu().detach() for l, v in net.state_dict().items()}))
            prev_state_dicts = prev_state_dicts[-5:]
            net.eval()
            tot = 0
            correct = 0
            for j, val_loader in enumerate(val_loaders):
                
                with torch.no_grad():
                    for item in val_loader:
                        ims = item[0].cuda()
                        labs = item[1].cuda()
                        preds = net(ims)
                        preds = torch.sigmoid(
                            preds).cpu().detach().numpy()
                        right = preds.argmax(
                            axis=1) == labs.cpu().detach().numpy()
                        tot += len(right)
                        correct += sum(right)
            val_accs.append(correct / tot)
            net.train()
            if len(val_accs) >= 5:
                cvs = val_accs[-5:]
                is_desc = True
                for i in range(4):
                    is_desc = is_desc and cvs[i] > cvs[i+1]
                if is_desc:
                    print("Early stopping!")
                    net.load_state_dict(prev_state_dicts[0])
                    net.cuda()
                    break


            for batch_n, data in enumerate(train_loaders[-1]):
                # get the inputs; data is a list of [inputs, labels]
                rl = 0.0
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()

                outputs = net(inputs)
                if task == 'EWC':
                    loss = criterion(outputs, labels, net)
                else:
                    loss = criterion(outputs, labels)
                rl += loss.item()
                loss.backward()
                optimizer.step()
                if batch_n % 100 == 99:
                    print("Loss: {}".format(round(rl / 1000, 6)))
        net.eval()
        tot = 0
        correct = 0
        for j, test_loader in enumerate(test_loaders):
            
            with torch.no_grad():
                for item in test_loader:
                    ims = item[0].cuda()
                    labs = item[1].cuda()
                    preds = net(ims)
                    preds = torch.sigmoid(
                        preds).cpu().detach().numpy()
                    right = preds.argmax(
                        axis=1) == labs.cpu().detach().numpy()
                    tot += len(right)
                    correct += sum(right)
        print("End of task test acc: {}%".format(round(100 * correct / tot, 3)))
        test_accs.append(correct / tot)
        net.train()
        if task == 'EWC':
            fishers.append(get_fisher(net, cur_trset))
            copy_net = Net()
            copy_net.load_state_dict(net.state_dict())
            pnets.append(copy_net.cuda())

    with open('{}_b.pickle'.format(task), 'wb') as f:
        pickle.dump(test_accs, f)


perts = [Scramble() for i in range(10)]

for task in ['EWC']:
    print("---- Starting {} ----".format(task))
    train(perts, task)
