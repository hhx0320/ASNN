import  torch
import torch.nn as nn



si=nn.Sigmoid()
def fenge():
    T=7
    n=[(i)/T for i in range(T)]

    n.reverse()
    print("开始了")
    total=100000
    max=(total/n.__len__())+100
    min=max-200

    for i in range(len(n)-1):
        while True:
            a = torch.randn(total)
            a = si(a)
            if i!=0:
                a = torch.where(a >= n[i-1], 0., a)
            sum=torch.where(a>=n[i],1.,0).sum()
            flag =0
            if sum>max:
                n[i]+=0.0001

            elif max>sum>min:

                break
            else:

                n[i] -= 0.0001
            # print("调整P{}为{}".format(i+1,n[i]))

        print(n)
    print(n)
fenge()
#
def erjinzhi():
    # a=torch.randn(100000)
    # a=si(a)
    # p=0
    # for i in a:
    #     if n[8]<i <n[7]:p+=1
    #
    leng=7
    lis=[[] for _ in range(leng)]
    for i in range(1,128):
        x=bin(i)[2:]

        if len(x) < leng:
             x=x.zfill(leng)
        for  n in range(leng):
            if x[n]=="1":
                # print(x[n])
                lis[n].append(i-1)

    for i in lis:
        print(i)
erjinzhi()
# #
# #
#
#
# a=torch.tensor((1,2,3))
# a=a.__reversed__()
# print(a)



def numworker():
    from time import time
    import multiprocessing as mp
    import torch
    import torchvision
    from torchvision import transforms
    transform = transforms.Compose([
        # transforms.Resize((40, 40), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(degrees=(0, 25)),
        transforms.RandomCrop((32,32),padding=5),
        transforms.ToTensor(),
                                    ])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform)

    print(f"num of CPU: {mp.cpu_count()}")
    for num_workers in range(0, mp.cpu_count(), 2):
        train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=num_workers, batch_size=128,
                                                   pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
# numworker()