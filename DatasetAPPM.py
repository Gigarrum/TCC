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
            #grid = grid - 0.5 #Normalizing data beetween -0.5 and 0.5
            
            return grid, label, resIdx
        
        else:
            
            #Stack channels to create 4D grid
            grid = torch.stack(channels,dim=-4) 
            return grid, label, resIdx
        
    def class_to_id(self,_class):
        return  torch.tensor(self.class_id[_class],dtype=torch.long)
        
    def id_to_class(self,_id):
        return self.id_class[_id]


#Aminoacid position propension matrix (APPM)
class DatasetAPPM_VLI(Dataset):

    class_id = {'VAL' : 0, 'LEU' : 1, 'ILE' : 2}

    id_class = {0 : 'VAL', 1 : 'LEU', 2 : 'ILE'}

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