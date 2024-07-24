
import random

import torch
import torch.nn as nn
from torch.nn import MaxPool2d, Sequential, Conv2d, ReLU
from torchvision.transforms import transforms


class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = 4.0
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            sgax = (ctx.saved_tensors[0] * ctx.alpha).sigmoid_()
            grad_x = grad_output * (1. - sgax) * sgax * ctx.alpha

        return grad_x, None


class Spike(nn.Module):
    def __init__(self):
        super(Spike, self).__init__()

        self.img_size = 32
        self.num_cls = 100
        self.num_steps = 7
        self.spike_fn = sigmoid.apply
        self.leak_mem = 0.7
        self.con1 = nn.Conv2d(3, 32, 3, padding=1)
        self.con11 = nn.Conv2d(3, 32, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn11 = nn.BatchNorm2d(32)


        self.con2 = nn.Conv2d(32, 48, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(48)

        self.con3 = nn.Conv2d(48, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.lin1 = nn.Linear(1024, 256)
        self.bn4 = nn.BatchNorm1d(256)

        self.lin2 = nn.Linear(256, self.num_cls)

        self.so = nn.Softmax(1)

        self.scale = nn.Parameter(torch.ones(self.num_steps))
        self.layers=4
        print("timestep={}".format(self.num_steps))

        self.leakelayers = nn.Parameter(torch.ones([self.layers,self.num_steps]) * 0.5)
        self.inputlayers = nn.Parameter(torch.ones([self.layers,self.num_steps]) * 0.5)
        
        self.lano = nn.LayerNorm(normalized_shape=[3, 32, 32])
        self.sigmoid = nn.Sigmoid()

    def forward(self, imgs,code,ASNN):
        max = MaxPool2d(2)

        mem_con1 = torch.zeros(imgs.shape[0], 32, 32, 32).cuda()
        mem_con2 = torch.zeros(imgs.shape[0], 48, 16, 16).cuda()
        mem_con3 = torch.zeros(imgs.shape[0], 64, 8, 8).cuda()

        mem_lin2 = torch.zeros(imgs.shape[0], 256).cuda()
        mem_fc1 = torch.zeros(self.num_steps, imgs.shape[0], self.num_cls).cuda()
        relu = nn.ReLU()
        lis = torch.zeros(self.num_steps, imgs.shape[0], 3, 32, 32).cuda()

        if code=="time":

            for i in range(self.num_steps):
                z = (i + 1.0)
                z = 1. - z / self.num_steps
                z = z.__float__()
                x = torch.where(imgs >= z, 1., 0.)
                imgs = imgs - imgs * x
                lis[i] = x
        elif code=="embedding":



            one=[0,2,4,6,8,10,12,14]
            two=[1,2,5,6,9,10,13,14]
            three=[3,4,5,6,11,12,13,14]
            four=[7,8,9,10,11,12,13,14]
            for i in range(15):
                z = (i + 1.0)
                z = 1. - z / self.num_steps
                z = z.__float__()
                x = torch.where(imgs >= z, 1., 0.)
                imgs = imgs - imgs * x
                if i in one:
                    lis[0] += x
                if i in two:
                    lis[1] += x
                if i in three:
                    lis[2] += x
                if i in four:
                    lis[3] += x
        elif code=="mix time":
            lis[0]=imgs
            for i in range(self.num_steps-1):
                z = (i + 1.0)
                z = 1. - z / self.num_steps
                z = z.__float__()
                x = torch.where(imgs >= z, 1., 0.)
                imgs = imgs - imgs * x
                lis[i+1] = x
        elif code=="mix embedding":

            one = [0, 2, 4, 6, 8, 10, 12, 14]
            two = [1, 2, 5, 6, 9, 10, 13, 14]
            three = [3, 4, 5, 6, 11, 12, 13, 14]
            four = [7, 8, 9, 10, 11, 12, 13, 14]
            lis[0]=imgs
            for i in range(15):
                z = (i + 1.0)
                z = 1. - z / self.num_steps
                z = z.__float__()
                x = torch.where(imgs >= z, 1., 0.)
                imgs = imgs - imgs * x
                if i in one:
                    lis[1] += x
                if i in two:
                    lis[2] += x
                if i in three:
                    lis[3] += x
                if i in four:
                    lis[4] += x

        elif code=="direct":

            lis=imgs.unsqueeze(0).repeat(self.num_steps, 1, 1, 1, 1)



        def lifnodelayer(cin, mem, t,layer):

            n = self.inputlayers[layer] [t] * cin
            mem = self.leakelayers[layer] [t] * mem + n

            mem_thr = mem  - 1
            x = self.spike_fn(mem_thr)
            mem = mem - x
            out = x.clone()
            return mem, out



        for t in range(self.num_steps):

            if code=="mix time" or code=="mix embedding" :
                if t==0:
                    x = self.bn11(self.con11(lis[t]))
                else:
                    x = self.bn1(self.con1(lis[t]))
            else:
                x = self.bn1(self.con1(lis[t]))
            mem_con1, out = lifnodelayer(x, mem_con1, t,0)
            

            # out_prev=max(out_prev)
            out = max(out)

            x = self.bn2(self.con2(out))
            mem_con2, out = lifnodelayer(x, mem_con2, t,1)
            out = max(out)

            x = self.bn3(self.con3(out))
            mem_con3, out = lifnodelayer(x, mem_con3, t,2)
            out = max(out)

            x_prev = nn.Flatten()(out)

            x = self.bn4(self.lin1(x_prev))

            mem_lin2, out = lifnodelayer(x, mem_lin2, t,3)
            x = self.lin2(out)

            # if(t<7):
            if ASNN == True:
                mem_fc1[t] = x * self.scale[t]
            else:
                mem_fc1[t] = x

        return mem_fc1.mean(0)
