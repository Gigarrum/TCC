import h5py
import helpers
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from Nets import VoxNet,MLP,DeeperVoxNet,VoxNet_VLI,MLP_VLI,TheNet
from DatasetAPPM import DatasetAPPM,DatasetAPPM_VLI
import sys

#Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Define net weights filename
#weights_filename = 'weights.pth'
weights_filename = sys.argv[2]

#Define epochs data filename
#epochs_data_filename = 'epochs_data.csv'
epochs_data_filename = sys.argv[3]

#Create neural network
# net = MLP_VLI()
net = TheNet(p=float(sys.argv[1]))
net.to(device)

#Define loss criterion
criterion = nn.CrossEntropyLoss()

#Define optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001) 
# optimizer = torch.optim.SGD(net.parameters(), lr=0.000001, momentum=0.9)

#Define number of training epochs
num_of_epochs = 100

#Define pré-training data transformations
transform = transforms.Compose(
    [transforms.ToTensor()])

#Load train set
print('Loading trainset...')

trainset = DatasetAPPM('/mnt/paulo_dbs/balanced_8000perClass_top8000_50hom_discretized_full_train60_20_20/train',transform = transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=0)

print('Trainset loaded.') 

#Load validation set
print('Loading validationset...')

validationset = DatasetAPPM('/mnt/paulo_dbs/balanced_8000perClass_top8000_50hom_discretized_full_train60_20_20/validation',transform = transform)

validationloader = torch.utils.data.DataLoader(validationset, batch_size=256,
                                         shuffle=False, num_workers=0)
print('Validationset loaded.')

#Train net
print('Training net...')

net.trainNet(trainloader,len(trainset),validationloader,len(validationset),epochs_data_filename,criterion,optimizer,num_of_epochs,device)

print('Training finished.')

#Validate net
print('Validating net...')

net.eval()
net.testNet(validationloader,device)

print('Net validated.')

#Save net
net.save(weights_filename)

print('Net saved.')

