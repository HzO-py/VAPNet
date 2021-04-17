import sys
import torch
from loss_optim import get_loss_optim
from load_dataset import VAPDataset,VAPTestDataset
from torch.utils.data import DataLoader
sys.path.append('../')
from model.vap_net import PNET,VANET,VAPNET
import matplotlib.pyplot as plt

gpus = [0]
vapdataset=VAPDataset()
vaptestdataset=VAPTestDataset()
dataloader = DataLoader(vapdataset, batch_size=32,shuffle=True)
test_dataloader = DataLoader(vaptestdataset, batch_size=32,shuffle=False)
#net=VANET(is_attention=True,is_coa=True)
net=VAPNET()
#net=PNET()
device = torch.device("cuda:4")
net.to(device)
epoch_num=200

x=[i for i in range(epoch_num)]
train_acc=[]
train_loss=[]
test_acc=[]

for epoch in range(epoch_num):
    total=0
    correct=0
    for i, data in enumerate(dataloader):
        videos,audios,poses,labels=data.values()
        videos=videos.to(device)
        audios=audios.to(device)
        poses=poses.to(device)
        labels=labels.to(device)

        outputs=net(videos,audios,poses)

        losscriterion,optimizer=get_loss_optim(net.parameters())
        optimizer.zero_grad()
        loss = losscriterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc.append(correct/total)
    train_loss.append(loss.item())

    total=0
    correct=0
    for i, data in enumerate(test_dataloader):
        videos,audios,poses,labels=data.values()
        videos=videos.to(device)
        audios=audios.to(device)
        poses=poses.to(device)
        labels=labels.to(device)

        outputs=net(videos,audios,poses)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    test_acc.append(correct/total)
    print(epoch,train_loss[-1],train_acc[-1],test_acc[-1])

title='VAP'
plt.title(title)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel('epoch')
plt.ylabel('%')
plt.plot(x, train_loss, marker='o', markersize=3)
plt.plot(x, train_acc, marker='o', markersize=3)
plt.plot(x, test_acc, marker='o', markersize=3)

plt.legend(['train_loss', 'train_acc', 'test_acc'])
plt.savefig('/hdd/sdd/lzq/dianwang/model/'+title+'.png')

    #torch.save(net.state_dict(), '/hdd/sdd/lzq/dianwang/model/'+str(loss.item())+'.pkl')
