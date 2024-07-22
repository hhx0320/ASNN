import torch
import torchvision.datasets
from torch import nn, randperm

from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, AvgPool2d, LPPool2d, Sequential, ReLU, BatchNorm2d
from torch.utils.data import DataLoader, random_split

from torchvision.transforms import transforms, InterpolationMode
from tqdm import tqdm

from lenet_spiking import Spike

class Me(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.con1=Conv2d(3, 32, 3, padding=1)
        self.bn1=nn.BatchNorm2d(32)
        self.con2 = nn.Conv2d(32, 48, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(48)
        self.con3 = nn.Conv2d(48, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.lin1 = nn.Linear(1024, 256)
        self.bn4 = nn.BatchNorm1d(256)

        self.lin2 = nn.Linear(256, 100)

    def forward(self,x):
        fl = Flatten()
        max=MaxPool2d(2)
        relu=nn.ReLU()
        x=relu(self.bn1(self.con1(x)))
        x=max(x)
        x=self.con2(x)
        x=relu(self.bn2(x))
        x=max(x)

        x = self.con3(x)
        x = relu(self.bn3(x))
        x = max(x)
        x=fl(x)
        # #
        x = self.lin1(x)
        x = relu(self.bn4(x))

        x = self.lin2(x)



        return  x
import numpy as np

code="mix time"#选择编码方式"time""mix time""direct"
ASNN=True#选择是否使用ASNN
if code=="time" or code=="embedding" or code=="mix time" or code=="mix embedding":
    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    ])
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(degrees=(0, 25)),
        transforms.RandomCrop((32,32),padding=5),
                                    transforms.ToTensor(),
                                    # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616])
                                    ])
    print("未使用normalize")
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616])
    ])
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(degrees=(0, 25)),
        transforms.RandomCrop((32, 32), padding=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616])
    ])
    print("使用normalize")

trainset=torchvision.datasets.CIFAR100(root="./data",train=True,transform=transform_train,download=True)

testset=torchvision.datasets.CIFAR100(root="./data",train=False,transform=transform,download=True)

test_data_size=len(testset)
train_data_size=len(trainset )




if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False,num_workers=0)


    # spike=Me().cuda()
    spike=Spike().cuda()#使用ANN时注释掉这行使用上面的
    n_parameters = sum(p.numel() for p in spike.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    loss_fn=nn.CrossEntropyLoss()
    loss_fn2=nn.MSELoss()
    loss_fn2=loss_fn2.cuda()
    loss_fn=loss_fn.cuda()
    learning_rate=0.005
    epoch = 200
    print("epoch:{}".format(epoch))
    optimizer=torch.optim.Adam(spike.parameters(),lr=learning_rate)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epoch,eta_min=1e-5)




    step = 0
    test_step=0
    max=MaxPool2d(2)
    avg=AvgPool2d(2)
    mix_off=145#mixup关闭轮数
    print("mix_off:{}".format(mix_off))
    print("编码方式{}，是否使用ASNN:{}".format(code,ASNN))
    print("mix up")
    for i in range(epoch):



        a = [["{:.4f}".format(nu) for nu in num ]for num in spike.leakelayers.tolist()]
        for l in a:
            print("mem_leake{}".format(l))
        a = [["{:.4f}".format(nu) for nu in num ] for num in spike.inputlayers.tolist()]
        for l in a:
            print("mem_input{}".format(l))
        a = [["{:.4f}".format(nu) for nu in num] for num in spike.re.tolist()]

        for l in a:
            print("re{}".format(l))


        a = ["{:.4f}".format(num) for num in spike.scale.tolist()]
        print("scale{}".format(a))

        print("第{}轮训练开始了".format(i+1))
        loss_b=0
        total_accuracy1=0

        if i == mix_off:
            print("mix off")

        lrl = [param_group['lr'] for param_group in optimizer.param_groups]
        print("lrl{}".format(lrl))
        progress_bar = tqdm(total=len(train_loader))
        for data in train_loader:
            imgs, targets = data
            if i<mix_off:
                index=torch.randperm(imgs.size(0))
                lam=torch.rand(1)
                imgs_a,imgs_b=imgs,imgs[index]
                targets_a,targets_b=targets,targets[index]
                mix_imgs=lam*imgs_a+(1-lam)*imgs_b
                lam=lam.cuda()
                mix_imgs = mix_imgs.cuda()
                targets_b = targets_b.cuda()
                targets_a = targets_a.cuda()
                output= spike(mix_imgs,code,ASNN)
                # output= spike(mix_imgs)
                accuracy = lam * (output.argmax(1) == targets_a).sum() + (1 - lam) * (output.argmax(1) == targets_b).sum()
                total_accuracy1 = total_accuracy1 + accuracy
                loss = lam * loss_fn(output, targets_a) + (1 - lam) * loss_fn(output, targets_b)

            # imgs=transform_train(imgs)

            else:
                targets = targets.cuda()
                imgs=imgs.cuda()

                output = spike(imgs,code,ASNN)
                # output = spike(imgs)

                loss = loss_fn(output, targets)
                accuracy = (output.argmax(1) == targets).sum()
                total_accuracy1 = total_accuracy1 + accuracy

            #
            #
            loss_b=loss_b+loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
        progress_bar.close()

        sch.step()

        print("这是第{}个epoch训练的损失".format(i+1),loss_b)
        print("训练集正确率{}".format(total_accuracy1/train_data_size))

        total_test_lost1=0
        total_test_lost2 = 0
        total_test_lost3 = 0
        total_test_lost4 = 0
        total_accuracy_test1=0
        total_accuracy_test2 = 0
        total_accuracy_test3 = 0
        total_accuracy_test4 = 0
        with torch.no_grad():

            n = 0
            for data in test_loader:
                imgs,targets=data
                targets = targets.cuda()
                imgs = imgs.cuda()
                output = spike(imgs,code,ASNN)
                # output = spike(imgs)
                loss = loss_fn(output, targets)

                total_test_lost1=total_test_lost1+loss
                accuracy = (output.argmax(1) == targets).sum()
                total_accuracy_test1 = total_accuracy_test1 + accuracy
        print("整体测试集上的loss：{}".format(total_test_lost1))
        print("整体测试集上的正确率{}".format(total_accuracy_test1 / test_data_size))







