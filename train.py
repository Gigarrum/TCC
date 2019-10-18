import h5py
import helpers
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from Nets import VoxNet,MLP,DeeperVoxNet,VoxNet_VLI
from DatasetAPPM import DatasetAPPM,DatasetAPPM_VLI

#Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Create neural network
net = VoxNet_VLI()
net.to(device)

#Define loss criterion
criterion = nn.CrossEntropyLoss()

#Define optimizer
#optimizer = torch.optim.Adam(net.parameters(), lr=0.01) #Testado com 0.001, 0.01, 0.1 N convergiu
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

#Define pré-training data transformations
transform = transforms.Compose(
    [transforms.ToTensor()])

#Load train set
print('Loading trainset...')

trainset = DatasetAPPM_VLI('D:/TCC/10%_top8000_50hom_discretized_V_L_I_train60_20_20/train',transform = transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

print('Trainset loaded.') 

#Load validation set
print('Loading validationset...')

validationset = DatasetAPPM_VLI('D:/TCC/10%_top8000_50hom_discretized_V_L_I_train60_20_20/validation',transform = transform)

validationloader = torch.utils.data.DataLoader(validationset, batch_size=4,
                                         shuffle=False, num_workers=0)
print('Validationset loaded.')

#Train net
print('Training net...')

net.trainNet(trainloader,criterion,optimizer,1,device)

print('Training finished.')

#Validate net
print('Validating net...')

net.eval()
net.testNet(validationloader,device)

print('Net validated.')

#Save net
net.save('weights.pth')

print('Net saved.')

