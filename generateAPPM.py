from DataTransformer import DataTransformer
from VoxNet import Net,DatasetAPPM
import torch
import torchvision.transforms as transforms

dataTransformer = DataTransformer(21,0.7,4)

dataTransformer.cleanDirectory('grids/')

#PASS THIS PARAMETERS BY ARGC/ARGV
_ = dataTransformer.transform('1a3aA.pdb','.','grids/')

#Loading model
model = Net()
model.load_state_dict(torch.load('weights.pth'))
model.eval()

#Transformation to be done with loaded data
transform = transforms.Compose(
    [transforms.ToTensor()])

#Load data for inference
inferset = DatasetAPPM('grids/',transform = transform)

#Create iterator for inference data
inferloader = torch.utils.data.DataLoader(inferset, batch_size=1,
                                         shuffle=False, num_workers=0)

#Make infereces and mount Aminoacid position propension matrix (APPM)
correct = 0
total = 0
APPM = []
model.eval()
with torch.no_grad():
    for data in inferloader:
        grids, labels, resIdx = data
        output = model(grids)
        APPM.append((resIdx,output))
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Network Accuracy: %d %%' % (
    100 * correct / total))

#Generate file with inference values????????? 
