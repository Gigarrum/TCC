from DataTransformer import DataTransformer
from Nets import TheNet
from DatasetAPPM import DatasetAPPM
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

dataTransformer = DataTransformer(21,0.7,4)

dataTransformer.cleanDirectory('grids/')


print('Transforming Data...')
#PASS THIS PARAMETERS BY ARGC/ARGV + weights.pth path
_ = dataTransformer.transform('1bpx.pdb','.','grids/')


#Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Loading model
net = TheNet(fc1_n_neurons=120)
net.load_state_dict(torch.load('C:/Users/pbmau/Documents/Paulo/Faculdade/TCC/System/experiments/TheNet-arch1/TheNet-drop=0,3-db=balanced10%-batch=256-Adam-lr=0,0001-epcs=100/weights.pth',map_location=torch.device('cpu')))
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


resSequence = []
predictions = []
APPM = []

softmax = nn.Softmax(dim=1)

with torch.no_grad():
    for data in inferloader:
        grids, labels, resIdx = data
        resIdx = resIdx[0]
        resSequence.append(labels.item())
        print(resSequence)
        output = net(grids)
        print(output.shape)
        output = softmax(output)
        print(output.shape)
        #output = output.view(-1)
        print(output.shape)
        predictions.append(output[0].numpy())
        print(str(resIdx) + ' : ', output)

print(len(resSequence),len(predictions))



for i in predictions:
    print(i)

print('Inferences finished.')

print('Generating Aminocid Propension per Position Matrix...')


predictions = np.asarray(predictions)
print(predictions.shape)

print(predictions)

plt.figure(figsize=(40,10))
ax = sns.heatmap(predictions.T,yticklabels=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V'])
#ax.set_ylim(len(predictions.T)-0.5, -0.5) #Bug correction for matplotlib 3.1.1 version
plt.savefig('Propensions.png')

print('Matrix Generated and Saved.')



