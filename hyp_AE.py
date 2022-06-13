
from turtle import forward
# from cv2 import mean
import torchvision.datasets as datasets
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F    
from torchvision.transforms import ToTensor
# from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer, required
from geoopt import optim
# optim.RiemannianAdam
# import hyptorchnn.Linear
# import hyptorch.nn as nn
# import hyptorch.nn.functional as F
import Hypmath
from hyp_lin import HypLinear, HypLinear1
# temp._expmap(test_set[0][0])
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = datasets.MNIST(root='/home/shivanand/Desktop/Minor/Datasets', train=True, transform=trans, download=True)
test_set =  datasets.MNIST(root='/home/shivanand/Desktop/Minor/Datasets', train=False, transform=trans, download=True)
batch_size = 100
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)
print(1-test_set[0][0])
print(test_set[0][0])
expmap=Hypmath._expmap
logmap=Hypmath._logmap0
prjct=Hypmath._project
vvv=expmap(test_set[0][0],0.5,0.5)
# print(expmap(test_set[0][0],0.5,0.5))        
# print(logmap(vvv,0.5))        
print(len(train_loader),len(test_loader))                
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
class hpy_ae(nn.Module):
    def __init__(self):
        super().__init__()
        # lst=[728,500,]
        self.dropout = nn.Dropout(p=0.1)
        self.e1=HypLinear(28*28,500)
        self.e2=HypLinear(500,2000)
        self.e3=HypLinear(2000,10)
        self.e4=HypLinear(10,10)
        self.d1=HypLinear(10,2000)
        self.d2=HypLinear(2000,500)
        self.d3=HypLinear(500,500)
        self.d4=HypLinear(500,784)
        self.rl=nn.ReLU()
    def forward(self,x):
        x=x.view(-1,1*28*28)
        x = self.dropout(x)
        x=expmap(x,x,0.9)
        x=self.rl(logmap(self.e1(x),0.9))
        x=expmap(x,x,0.9)
        x=self.rl(logmap(self.e2(x),0.9))
        x=expmap(x,x,0.9)
        x=self.rl(logmap(self.e3(x),0.9))
        x=expmap(x,x,0.9)
        emb=self.rl(logmap(self.e4(x),0.9))
        emb=expmap(emb,x,0.9)
        dx=self.rl(logmap(self.d1(emb),0.9))
        dx=expmap(dx,dx,0.9)
        dx=self.rl(logmap(self.d2(dx),0.9))
        dx=expmap(dx,dx,0.9)
        dx=self.rl(logmap(self.d3(dx),0.9))
        dx=expmap(dx,dx,0.9)
        dx=(self.d4(dx))
        dx=dx.view(-1,1,28,28)
        # dx=logmap(dx,0.9)
        return emb,dx
model=hpy_ae().to(device)        
lst=[]
def hyp_train(train_loader, epochs):
        # dec_ae = torch.load("/content/drive/MyDrive/Minor_models/proxymnist")
        

        # model=torch.load('/home/shivanand/Desktop/Minor/codes/hyp_ae1.pt')
        mseloss = nn.MSELoss()
        optimizer =optim.RiemannianAdam(model.parameters(),
                                 lr=0.01)
        best_acc = 0.0
        for epoch in range(epochs):
            model.train()
            running_loss=0.0
            for i,(data,label )in enumerate(train_loader):
                # model=torch.load('/home/shivanand/Desktop/Minor/codes/hyp_ae1.pt')
                data=1-data.to(device)
                hdata=Hypmath._expmap0(data,0.9)
                hdata=Hypmath._project(hdata,0.9)
                # print(data.size())
                label=label.to(device)
                optimizer.zero_grad()
                _,x_de1 = model(data)
                # print(x_de.size())
                lst.append(x_de1)
                x_de=logmap(x_de1,0.9)
                loss1 = mseloss(x_de1,data) 
                # loss2=F.mse_loss(x_de1,x_de,reduce=True)
                loss=loss1
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * data.size(0)
                if i % 100 == 99:    # print every 100 mini-batches
                    print('[%d, %5d] loss: %.7f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
            #now we evaluate the accuracy with AE
            # dec_ae.eval()
            # currentAcc =self.validateOnCompleteTestData(test_loader,dec_ae)
            # if currentAcc > best_acc:                
            torch.save(model,'/content/drive/MyDrive/Minor/hypmnist.pt')
                                                                                                                              
# hyp_train(train_loader,200)
# model=torch.load('/home/shivanand/Desktop/Minor/codes/hypmnist.pt',map_location=torch.device('cpu'))
model=torch.load('/home/shivanand/Desktop/Minor/codes/hyp_ae1.pt')

# # # # lst=[]
# # # # for (data,lab) in test_loader:
# # # #       emb,_=model(data)
# # # #       emb=emb.detach().cpu().numpy()
# # # #       lst.extend(emb)
# # # # lst=np.array(lst)
# # # # np.save("/home/shivanand/Desktop/Minor/codes/tmp.npy",lst)      
# # # import geoopt
# # # from geoopt import PoincareBall
# # # print(PoincareBall.check_point_on_manifold(self=PoincareBall,x=vvv))
emb,tmp=model(1-test_set[10][0])
print(tmp.shape,emb.shape)
tmp=(tmp.detach().cpu().numpy()).squeeze(0)
g=prjct(expmap((1-test_set[10][0]),0.9,2),2)

# print(test_set[1000][0])
plt.imshow(g.squeeze(0))
plt.show()
# l=logmap(g,0.9)
# plt.imshow(l.squeeze(0))
# plt.show()
plt.imshow((1-test_set[10][0]).squeeze(0))
plt.show()
plt.imshow(tmp.squeeze(0))
plt.show()
# # import geomstats
# from geomstats.geometry import riemannian_metric
# riemannian_metric.RiemannianMetric.dist
