from pathlib import Path
import torch
from torch import Tensor
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import h5py
import time

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 4 input image channel, 6 output channels, 3x3x3 square convolution
        # kernel
        self.conv1 = nn.Conv3d(4, 4, 3)
        self.conv2 = nn.Conv3d(4, 4, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(4 * 13 * 13 * 13, 120) 
        self.fc2 = nn.Linear(120, 20)
    
    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        # Max pooling over a (2,2,2) window
        x = F.max_pool3d(F.relu(self.conv2(x)), (2,2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
             num_features *= s
        return num_features

    def trainNet(self,trainloader,criterion,optimizer,epochs,device):

        for epoch in range(epochs):  # loop over the dataset multiple times

            #Benchmark
            startTime = time.time()

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                images, labels, _ = data

                # send data to device
                images = images.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                #if i % 10 == 9:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 1))
                running_loss = 0.0

            #Benchmark
            endTime = time.time()
            print('Epoch Total time: ',endTime - startTime)

        print('Finished Training')

    def testNet(self,testloader,device):

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels, _ = data

                # send data to device
                images = images.to(device)
                labels = labels.to(device)

                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Network Accuracy: %d %%' % (
            100 * correct / total))

    def save(self,weight_filename):
        torch.save(self.state_dict(), weight_filename)

    def load(self,weight_filename):
        self.load_state_dict(torch.load(weight_filename))
        self.eval()

#Aminoacid position propension matrix (APPM)
class DatasetAPPM(Dataset):
    
    class_id = {'ALA' : 0, 'ARG' : 1, 'ASN' : 2, 'ASP' : 3, 'CYS' : 4, 'GLU' : 5,
           'GLN' : 6, 'GLY' : 7, 'HIS' : 8, 'ILE' : 9, 'LEU' : 10, 'LYS' : 11,
           'MET' : 12, 'PHE' : 13, 'PRO' : 14, 'SER' : 15, 'THR' : 16, 'TRP' : 17, 'TYR' : 18,
           'VAL' : 19}
    id_class = {0 : 'ALA', 1 : 'ARG', 2 : 'ASN', 3 : 'ASP', 4 : 'CYS', 5 : 'GLU',
          6 : 'GLN', 7 : 'GLY', 8 : 'HIS', 9 : 'ILE', 10 : 'LEU', 11 : 'LYS',
          12 : 'MET', 13 : 'PHE', 14 : 'PRO', 15 : 'SER', 16 : 'THR', 17 : 'TRP', 18 : 'TYR',
          19 : 'VAL'}
    
    def __init__(self, dir_path, transform=None):
        
        #Store database file paths
        self.data = []
        self.transform = transform
        
        #Recover file paths from directory
        p = Path(dir_path)
        assert(p.is_dir())
        self.data = sorted(p.glob('*.hdf5'))
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        #Recover grid from dataset and label from filename
        with h5py.File(self.data[index]) as h5_file:
            resIdx = str(self.data[index]).split('_')[1]
            label = str(self.data[index]).split('_')[-1].split('.')[0]
            label = self.class_to_id(label)
            channels = list(h5_file['grid'])
            
        
        if self.transform is not None:
            
            transf_channels = []
            
            #Apply transformations on each grid channel
            for channel in channels: 
                transf_channels.append(self.transform(channel))
                
            #Stack transformed channels to create 4D grid
            grid = torch.stack(transf_channels,dim=-4)
            grid = grid.float()
            grid = grid - 0.5 #Normalizing data beetween -0.5 and 0.5
            
            return grid, label, resIdx
        
        else:
            
            #Stack channels to create 4D grid
            grid = torch.stack(channels,dim=-4) 
            return grid, label, resIdx
        
    def class_to_id(self,_class):
        return  torch.tensor(self.class_id[_class],dtype=torch.long)
        
    def id_to_class(self,_id):
        return self.id_class[_id]