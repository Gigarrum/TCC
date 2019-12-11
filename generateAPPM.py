from DataTransformer import DataTransformer
from Nets import TheNet
from DatasetAPPM import DatasetAPPM
import torch
import torchvision.transforms as transforms

dataTransformer = DataTransformer(21,0.7,4)

dataTransformer.cleanDirectory('grids/')

#PASS THIS PARAMETERS BY ARGC/ARGV + weights.pth path
_ = dataTransformer.transform('1bpx.pdb','.','grids/')


#Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Loading model
net = TheNet(fc1_n_neurons=120)
net.load_state_dict(torch.load('C:/Users/pbmau/Documents/Paulo/Faculdade/TCC/System/experiments/TheNet-arch1/TheNet-drop=0,3-db=balanced10%-batch=256-Adam-lr=0,0001-epcs=100/weights.pth'))
net.eval()

net.to(device)

#Transformation to be done with loaded data
transform = transforms.Compose(
    [transforms.ToTensor()])

#Load data for inference
inferset = DatasetAPPM('grids/',transform = transform)

#Create iterator for inference data
inferloader = torch.utils.data.DataLoader(inferset, batch_size=1,
                                         shuffle=False, num_workers=0)

APPM = []

with torch.no_grad():
    for data in inferloader:
        grids, labels, resIdx = data
        output = net(grids)
        APPM.append((resIdx,output))

print(AAPM)

