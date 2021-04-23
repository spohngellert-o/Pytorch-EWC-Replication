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
import pdb
from utils import *


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 10)
        )

    def forward(self, x):
        return self.model(x.float())


def train(perts, lf):
    net = Net()
    net.cuda()
    pnets = []
    fishers = []
    train_loaders = []
    test_loaders = []
    test_accs = []
    for i, pert in enumerate(perts):
        print("Starting task {}".format(i+1))
        cur_trset = datasets.MNIST('./files', train=True, transform=transforms.Compose([
            pert,
            transforms.ToTensor()
        ]))
        cur_tset = testset1 = datasets.MNIST('./files', train=False, transform=transforms.Compose([
            pert,
            transforms.ToTensor()
        ]))
        train_loaders.append(torch.utils.data.DataLoader(
            cur_trset,
            batch_size=1,
            num_workers=2,
            drop_last=False,
            shuffle=True))

        test_loaders.append(torch.utils.data.DataLoader(
            cur_tset,
            batch_size=50,
            num_workers=2,
            drop_last=False))
        test_accs.append([])
        if lf == 'EWC':
            criterion = EWC(fishers, pnets, 0.01)
        elif lf == 'L2':
            criterion = L2(pnets, 0.5)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(net.parameters(), lr=10 ** -3)
        for epoch in range(20):  # loop over the dataset multiple times
            print("Starting epoch {}".format(epoch))

            for batch_n, data in enumerate(train_loaders[-1]):
                # get the inputs; data is a list of [inputs, labels]
                rl = 0.0
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()

                outputs = net(inputs)
                if lf in ['EWC', 'L2']:
                    loss = criterion(outputs, labels, net)
                else:
                    loss = criterion(outputs, labels)
                rl += loss.item()
                loss.backward()
                optimizer.step()
                if batch_n % 1000 == 999:
                    print("Loss: {}".format(round(rl / 1000, 6)))
             
                if batch_n % 10000 == 9999:
                    net.eval()
                    for j, test_loader in enumerate(test_loaders):
                        tot = 0
                        correct = 0
                        with torch.no_grad():
                            for item in test_loader:
                                ims = item[0].cuda()
                                labs = item[1].cuda()
                                preds = net(ims)
                                preds = torch.sigmoid(preds).cpu().detach().numpy()
                                right = preds.argmax(
                                    axis=1) == labs.cpu().detach().numpy()
                                tot += len(right)
                                correct += sum(right)
                        test_accs[j].append(correct / tot)
                        print("Curr test acc for task {}: {}".format(j+1, correct / tot))
                    net.train()
        if lf == 'EWC':
            fishers.append(get_fisher(net, cur_trset))
            copy_net = Net()
            copy_net.load_state_dict(net.state_dict())
            pnets.append(copy_net.cuda())

        elif lf == 'L2':
            copy_net = Net()
            copy_net.load_state_dict(net.state_dict())
            pnets.append(copy_net.cuda())

    with open('{}_a.pickle'.format(lf), 'wb') as f:
        pickle.dump(test_accs, f)



perts = [Scramble() for i in range(3)]

for lf in ['L2']:
    print("---- Starting {} ----".format(lf))
    train(perts, lf)

