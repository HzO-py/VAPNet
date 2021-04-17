import torch.nn as nn
from torch.optim import Adam




def get_loss_optim(parameters):
    loss=nn.CrossEntropyLoss()
    optimizer=Adam(parameters,lr=0.001,betas=(0.9,0.99))
    return loss,optimizer