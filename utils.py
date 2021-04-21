import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from torch.autograd import Variable


class Scramble(object):

    def __init__(self):
        self.seed = np.random.randint(10 ** 8)

    def __call__(self, pic):
        rng = np.random.default_rng(self.seed)
        pic = np.array(pic)
        rng.shuffle(pic.flat)
        pic = (pic / 255).astype(float)
        return pic


    def __repr__(self):
        return self.__class__.__name__ + '()'

class ScrambleSize(object):

    def __init__(self, size):
        self.seed = np.random.randint(10 ** 8)
        self.size = size // 2

    def __call__(self, pic):
        rng = np.random.default_rng(self.seed)
        pic = np.array(pic)
        rng.shuffle(pic[14 - self.size:14 + self.size, 14 - self.size:14 + self.size].flat)
        pic = (pic / 255).astype(float)
        return pic

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_fisher(net, tset):
    tset = torch.utils.data.DataLoader(
        tset,
        batch_size=1,
        num_workers=2,
        drop_last=False,
        shuffle=True)
    net.eval()
    lf = nn.CrossEntropyLoss()
    sums = [torch.zeros(tuple(param.shape)).to('cuda:0')
            for param in net.parameters()]
    for pic, lab in tqdm(tset, desc="Calculating fisher matrix"):
        out = net(pic.cuda())
        loss = lf(out, lab.cuda())
        net.zero_grad()
        loss.backward()

        for i, param in enumerate(net.parameters()):
            sums[i] += param.grad.detach() ** 2
    net.train()
    return sums


class EWC(nn.Module):

    def __init__(self, fishers, pnets, lam):
        super(EWC, self).__init__()
        self.fishers = fishers
        self.pnets = pnets
        self.lam = lam
        self.lf = nn.CrossEntropyLoss()

    def forward(self, outputs, labels, net):
        reg = Variable(torch.tensor([0.]), requires_grad=True).to('cuda:0')
        for fis, pnet in zip(self.fishers, self.pnets):
            for cf, p1, p2 in zip(fis, net.parameters(), pnet.parameters()):
                reg = reg + (cf * (p1 - p2)).norm(2)
        return self.lf(outputs, labels) + self.lam * reg


class L2(nn.Module):

    def __init__(self, pnets, lam):
        super(L2, self).__init__()
        self.pnets = pnets
        self.lam = lam
        self.lf = nn.CrossEntropyLoss()

    def forward(self, outputs, labels, net):
        reg = Variable(torch.tensor([0.]), requires_grad=True).to('cuda:0')
        for pnet in self.pnets:
            for p1, p2 in zip(net.parameters(), pnet.parameters()):
                reg = reg + ((p1 - p2)).norm(2)

        return self.lf(outputs, labels) + self.lam * reg