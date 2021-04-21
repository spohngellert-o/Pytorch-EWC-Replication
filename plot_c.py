import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
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
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.model(x.float())


def train(size):
    net = Net()
    net.cuda()
    pnets = []
    fishers = []
    train_loaders = []
    perts = [ScrambleSize(size) for i in range(2)]
    for i, pert in enumerate(perts):
        print("Starting task {}".format(i + 1))
        cur_trset = datasets.MNIST('./files', train=True, transform=transforms.Compose([
            pert,
            transforms.ToTensor()
        ]))
        cur_tset = datasets.MNIST('./files', train=False, transform=transforms.Compose([
            pert,
            transforms.ToTensor()
        ]))
        train_loaders.append(torch.utils.data.DataLoader(
            cur_trset,
            batch_size=5,
            num_workers=2,
            drop_last=False,
            shuffle=True))

        test_loader = torch.utils.data.DataLoader(
            cur_tset,
            batch_size=50,
            num_workers=2,
            drop_last=False)
        
        criterion = EWC(fishers, pnets, 0.01)
        
        optimizer = torch.optim.SGD(net.parameters(), lr=10 ** -3)
        for epoch in range(100):  # loop over the dataset multiple times
            print("Starting epoch {}".format(epoch))
            net.eval()
            tot = 0
            correct = 0
                
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
            print("Current test acc: {}%".format(round(100 * correct / tot, 3)))
            net.train()
            for batch_n, data in enumerate(train_loaders[-1]):
                # get the inputs; data is a list of [inputs, labels]
                rl = 0.0
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels, net)
                rl += loss.item()
                loss.backward()
                optimizer.step()
                if batch_n % 200 == 199:
                    print("Loss: {}".format(round(rl / 1000, 6)))

        
        fishers.append(get_fisher(net, cur_trset))
        copy_net = Net()
        copy_net.load_state_dict(net.state_dict())
        pnets.append(copy_net.cuda())

    with open('{}_c.pickle'.format(size), 'wb') as f:
        pickle.dump(fishers, f)



for size in [26, 8]:
    print("---- Starting size {} ----".format(size))
    train(size)
