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

class TheNet(nn.Module):

    def __init__(self,p=0.0):
        super(TheNet, self).__init__()
        self.p = p
        
        #Convolution Layers
        self.conv1 = nn.Conv3d(4, 6, 7)
        self.conv2 = nn.Conv3d(6, 16, 7)
        self.conv3 = nn.Conv3d(16, 16, 7)
        self.conv4 = nn.Conv3d(16, 16, 7)

        #Fully connected layers
        self.fc1 = nn.Linear(16 * 6 * 6 * 6, 120)  
        self.fc2 = nn.Linear(120, 20)

        #Dropout layers Definition
        self.drop1 = nn.Dropout(p=self.p)
    
    def forward(self, x):
        x = (F.leaky_relu(self.conv1(x)))
        x = (F.leaky_relu(self.conv2(x)))
        x = (F.leaky_relu(self.conv3(x)))
        x = (F.leaky_relu(self.conv4(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.drop1(self.fc1(x))) #Dropout layer after activation, unless it's ReLu(Can acquire better performance)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
             num_features *= s
        return num_features

    def trainNet(self,trainloader,trainset_length,validationloader,validationset_length,filename,criterion,optimizer,epochs,device):

        #Print Net configuration
        print(self)

        #Write csv file header
        header = ','.join(['epoch','trainLoss','trainAccuracy','validationLoss','validationAccuracy'])
        with open(filename, 'a') as f:
                f.write(header + '\n')

                

        for epoch in range(epochs):  # loop over the dataset multiple times
            
            #Restart loss handlers values
            train_epoch_loss = 0
            validation_epoch_loss = 0

            #Start Benchmark timer
            startTime = time.time()

            #Enter training mode
            self.train()

            running_loss = 0.0
            correct = 0
            total = 0

            #--------------------------Train loop------------------------------
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                images, labels, _ = data

                # send data to device
                images = images.to(device)
                labels = labels.to(device)

                print(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(images)

                #Calculate epoch train accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                print(F.softmax(outputs))
                print(torch.argmax(F.softmax(outputs), dim=1))
                print(loss)
                loss.backward()
                optimizer.step()

                #Increment epoch loss value
                train_epoch_loss += outputs.shape[0] * loss.item()

                # print statistics
                running_loss += loss.item()
                #if i % 10 == 9:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 1))
                running_loss = 0.0

            #-------------------------------------------------------------------

            #Finish Benchmark timer
            print('---------------Epoch ' + str(epoch+1) + '---------------') 
            endTime = time.time()
            print('Total time:',endTime - startTime)
            
            #Finish TRAIN epoch loss calculation
            train_epoch_loss = train_epoch_loss / trainset_length
            #Print TRAIN epoch loss
            print('Train loss:', train_epoch_loss)

            #Calculate epoch TRAIN accuracy
            train_epoch_accuracy = (100 * correct / total)
            print('Train Accuracy:', train_epoch_accuracy, '%')

            #Enter evaluation mode
            self.eval()

            correct = 0
            total = 0

            #--------------------------Validation Loop--------------------------
            for data in validationloader:
                images, labels, _ = data

                # send data to device
                images = images.to(device)
                labels = labels.to(device)

                #Calculate validation loss
                outputs = self(images)
                loss = criterion(outputs, labels)

                #Increment epoch loss value
                validation_epoch_loss += outputs.shape[0] * loss.item()

                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            #---------------------------------------------------------------------

            #Finish VALIDATION epoch loss calculation
            validation_epoch_loss = validation_epoch_loss / validationset_length

            #print VALIDATION epoch loss
            print('Validation loss:',validation_epoch_loss)

            #Calculate epoch VALIDATION accuracy
            validation_epoch_accuracy = (100 * correct / total)
            print('Validation Accuracy:', validation_epoch_accuracy, '%')

            print('--------------------------------------------------------') 

            #Save epoch data on file for future analysis
            epoch_data = ','.join([str(epoch+1),str(train_epoch_loss),str(train_epoch_accuracy),str(validation_epoch_loss),str(validation_epoch_accuracy)])
    
            with open(filename, 'a') as f:
                f.write(epoch_data + '\n')


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

        print('Network Total Accuracy: %d %%' % (
            100 * correct / total))

        try:
            class_correct = list(0. for i in range(10))
            class_total = list(0. for i in range(10))
            with torch.no_grad():
                for data in testloader:
                    images, labels, _ = data
                    outputs = self(images)
                    _, predicted = torch.max(outputs, 1)
                    c = (predicted == labels).squeeze()
                    for i in range(4):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1

            for i in range(10):
                print('Accuracy of %5s : %2d %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))
        except Exception as ex:
            print('Test for each class failed by following exception: ',ex)


    def save(self,weight_filename):
        torch.save(self.state_dict(), weight_filename)

    def load(self,weight_filename):
        self.load_state_dict(torch.load(weight_filename))
        self.eval()


class VoxNet(nn.Module):

    def __init__(self):
        super(VoxNet, self).__init__()
        # 4 input image channel, 6 output channels, 3x3x3 square convolution
        # kernel
        self.conv1 = nn.Conv3d(4, 6, 3)
        self.conv2 = nn.Conv3d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 13 * 13 * 13, 120) 
        self.fc2 = nn.Linear(120, 20)
    
    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        # Max pooling over a (2,2,2) window
        x = F.max_pool3d(F.relu(self.conv2(x)), (2,2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
             num_features *= s
        return num_features

    def trainNet(self,trainloader,criterion,optimizer,epochs,device):

        for epoch in range(epochs):  # loop over the dataset multiple times
            print(self)

            #Benchmark
            startTime = time.time()

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                images, labels, _ = data

                # send data to device
                images = images.to(device)
                labels = labels.to(device)

                print(labels)

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

        print('Network Total Accuracy: %d %%' % (
            100 * correct / total))

        try:
            class_correct = list(0. for i in range(10))
            class_total = list(0. for i in range(10))
            with torch.no_grad():
                for data in testloader:
                    images, labels, _ = data
                    outputs = self(images)
                    _, predicted = torch.max(outputs, 1)
                    c = (predicted == labels).squeeze()
                    for i in range(4):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1

            for i in range(10):
                print('Accuracy of %5s : %2d %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))
        except Exception as ex:
            print('Test for each class failed by following exception: ',ex)


    def save(self,weight_filename):
        torch.save(self.state_dict(), weight_filename)

    def load(self,weight_filename):
        self.load_state_dict(torch.load(weight_filename))
        self.eval()

class DeeperVoxNet(nn.Module):

    def __init__(self):
        super(DeeperVoxNet, self).__init__()
        # 4 input image channel, 6 output channels, 3x3x3 square convolution
        # kernel
        self.conv1 = nn.Conv3d(4, 6, 3)
        self.conv2 = nn.Conv3d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 13 * 13 * 13, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 20)
    
    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        # Max pooling over a (2,2,2) window
        x = F.max_pool3d(F.relu(self.conv2(x)), (2,2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
             num_features *= s
        return num_features

    def trainNet(self,trainloader,criterion,optimizer,epochs,device):

        for epoch in range(epochs):  # loop over the dataset multiple times
            print(self)
            
            #Benchmark
            startTime = time.time()

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                images, labels, _ = data

                # send data to device
                images = images.to(device)
                labels = labels.to(device)

                #print(labels)

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

        print('Network Total Accuracy: %d %%' % (
            100 * correct / total))

        try:
            class_correct = list(0. for i in range(10))
            class_total = list(0. for i in range(10))
            with torch.no_grad():
                for data in testloader:
                    images, labels, _ = data
                    outputs = self(images)
                    _, predicted = torch.max(outputs, 1)
                    c = (predicted == labels).squeeze()
                    for i in range(4):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1

            for i in range(10):
                print('Accuracy of %5s : %2d %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))
        except Exception as ex:
            print('Test for each class failed by following exception: ',ex)


    def save(self,weight_filename):
        torch.save(self.state_dict(), weight_filename)

    def load(self,weight_filename):
        self.load_state_dict(torch.load(weight_filename))
        self.eval()



class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(30 * 30 * 30 * 4, 15 * 15 * 15 * 4) 
        self.fc2 = nn.Linear(15 * 15 * 15 * 4, 7 * 7 * 7 * 4)
        self.fc3 = nn.Linear(7 * 7 * 7 * 7, 20)
    
    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
             num_features *= s
        return num_features

    def trainNet(self,trainloader,criterion,optimizer,epochs,device):

        for epoch in range(epochs):  # loop over the dataset multiple times
            print(self)

            #Benchmark
            startTime = time.time()

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                images, labels, _ = data

                # send data to device
                images = images.to(device)
                labels = labels.to(device)

                print(labels)

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

        print('Network Total Accuracy: %d %%' % (
            100 * correct / total))

        try:
            class_correct = list(0. for i in range(10))
            class_total = list(0. for i in range(10))
            with torch.no_grad():
                for data in testloader:
                    images, labels, _ = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = self(images)
                    _, predicted = torch.max(outputs, 1)
                    c = (predicted == labels).squeeze()
                    for i in range(4):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1

            for i in range(10):
                print('Accuracy of %5s : %2d %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))
        except Exception as ex:
            print('Test for each class failed by following exception: ',ex)


    def save(self,weight_filename):
        torch.save(self.state_dict(), weight_filename)

    def load(self,weight_filename):
        self.load_state_dict(torch.load(weight_filename))
        self.eval()



class VoxNet_VLI(nn.Module):

    def __init__(self):
        super(VoxNet_VLI, self).__init__()
        # 4 input image channel, 6 output channels, 3x3x3 square convolution
        # kernel
        self.conv1 = nn.Conv3d(4, 6, 3)
        self.conv2 = nn.Conv3d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 13 * 13 * 13, 120) 
        self.fc2 = nn.Linear(120, 3)
    
    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        # Max pooling over a (2,2,2) window
        x = F.max_pool3d(F.relu(self.conv2(x)), (2,2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
             num_features *= s
        return num_features

    def trainNet(self,trainloader,criterion,optimizer,epochs,device):

        for epoch in range(epochs):  # loop over the dataset multiple times
            print(self)

            #Benchmark
            startTime = time.time()

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                images, labels, _ = data

                # send data to device
                images = images.to(device)
                labels = labels.to(device)

                print(labels)

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

        print('Network Total Accuracy: %d %%' % (
            100 * correct / total))

        try:
            class_correct = list(0. for i in range(10))
            class_total = list(0. for i in range(10))
            with torch.no_grad():
                for data in testloader:
                    images, labels, _ = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = self(images)
                    _, predicted = torch.max(outputs, 1)
                    c = (predicted == labels).squeeze()
                    for i in range(4):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1

            for i in range(10):
                print('Accuracy of %5s : %2d %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))
        except Exception as ex:
            print('Test for each class failed by following exception: ',ex)


    def save(self,weight_filename):
        torch.save(self.state_dict(), weight_filename)

    def load(self,weight_filename):
        self.load_state_dict(torch.load(weight_filename))
        self.eval()


class MLP_VLI(nn.Module):

    def __init__(self):
        super(MLP_VLI, self).__init__()
        
        self.fc1 = nn.Linear(30 * 30 * 30 * 4, 1024) 
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024,128)
        self.fc5 = nn.Linear(128, 3)
    
    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
             num_features *= s
        return num_features

    def trainNet(self,trainloader,criterion,optimizer,epochs,device):

        for epoch in range(epochs):  # loop over the dataset multiple times
            print(self)

            #Benchmark
            startTime = time.time()

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                images, labels, _ = data

                # send data to device
                images = images.to(device)
                labels = labels.to(device)

                print(labels)

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

        print('Network Total Accuracy: %d %%' % (
            100 * correct / total))

        try:
            class_correct = list(0. for i in range(10))
            class_total = list(0. for i in range(10))
            with torch.no_grad():
                for data in testloader:
                    images, labels, _ = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = self(images)
                    _, predicted = torch.max(outputs, 1)
                    c = (predicted == labels).squeeze()
                    for i in range(4):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1

            for i in range(10):
                print('Accuracy of %5s : %2d %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))
        except Exception as ex:
            print('Test for each class failed by following exception: ',ex)


    def save(self,weight_filename):
        torch.save(self.state_dict(), weight_filename)

    def load(self,weight_filename):
        self.load_state_dict(torch.load(weight_filename))
        self.eval()
