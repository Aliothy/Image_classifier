import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as opt
import time
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


class LoadedDataset(Dataset):

    def __init__(self, set):
        self.data = set['data']
        self.transform = set['transform']
        self.labels = set['labels']

    def __len__(self):
        return len(self.labels)
    
    #to obtain an item the directory and the name are used togheter, the image is found an opened
    def __getitem__(self, idx):
        image = self.data[:,:,:,idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
            image = image.float()
        return image,label

#definition fo the NN

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1, padding=0)
        self.maxp1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=7, kernel_size=3, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=7, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(3 * 3 * 4, 18)  
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(18, 9)  
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = self.conv3(x)
        x = self.maxp3(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x


if not(os.path.exists('training_set.pt')):

    # List all files needed
    file_names = os.listdir('output')
    #sort alphabetically to create the database
    file_names.sort()

    train_names = []
    test_names = []
    train_labels = np.zeros(72000)
    test_labels = np.zeros(18000)

    #since the dataset is alphabetically sorted, for each class the first 2000 items
    #will go under the test dataset and the other under the train
    for i in range(9):
        train_names = train_names+file_names[10000*i+2000:10000*(i+1)]
        train_labels[8000*i:8000*(i+1)] = i*np.ones(8000)
        test_names = test_names+file_names[10000*i:10000*i+2000]
        test_labels[2000*i:2000*(i+1)] = i*np.ones(2000)
   
    #let's shuffle them
    indeces = np.arange(72000)
    indeces = shuffle(indeces).astype(int)

    train_names = np.array(train_names)
    train_names = train_names[indeces]
    train_labels = train_labels[indeces]

    indeces = np.arange(18000)
    indeces = shuffle(indeces).astype(int)

    test_names = np.array(test_names)
    test_names = test_names[indeces]
    test_labels = test_labels[indeces]

    #let's create the two sets
    train_set = np.zeros([40,40,3,72000])
    test_set = np.zeros([40,40,3,18000])
    
    for i in range(72000):
        path = os.path.join('output', train_names[i] )
        image = cv2.imread(path)
        image = cv2.resize(image, [40,40], interpolation = cv2.INTER_AREA)
        image = (image/127.5)-1.0
        train_set[:,:,:,i]=image

    for i in range(18000):
        path = os.path.join('output', test_names[i] )
        image = cv2.imread(path)
        image = cv2.resize(image, [40,40], interpolation = cv2.INTER_AREA)
        image = (image/127.5)-1.0
        test_set[:,:,:,i]=image
    
    #let's create the two datasets
    training_data = {
        'data': train_set,
        'labels': train_labels,
        'transform': transforms.ToTensor()
    }

    torch.save(training_data, 'training_set.pt',pickle_protocol=4)

    testing_data = {
        'data': test_set,
        'labels': test_labels,
        'transform': transforms.ToTensor()
    }

    torch.save(testing_data, 'testing_set.pt',pickle_protocol=4)
 
#let's load the two datasets and train the NN
train = LoadedDataset(torch.load('training_set.pt'))
test = LoadedDataset(torch.load('testing_set.pt'))

#let's define the NN
NN_model = CNN() 
print(NN_model)

#define the optimizer
loss = nn.MSELoss()
epoch_number = 40
learn = 0.002
MSE_train = np.zeros(epoch_number)
acc_train = np.zeros(epoch_number)
MSE_test = np.zeros(epoch_number)
acc_test = np.zeros(epoch_number)

#let's train the network
for epoch in range(epoch_number):

    optimizer = opt.SGD(NN_model.parameters(), lr = learn, momentum = 0.8)

    #initialize the MSE of the train and the set to 0
    MSE_r_train = 0.0
    MSE_r_test = 0.0
    acc_r_train = 0
    acc_r_test = 0

    t0 = time.time()
    for i in range(len(train)):
        inputs, labels = train[i]
        #let's clear the gradient of all tensors
        optimizer.zero_grad()
        #evaluate the outputs
        outputs = NN_model(inputs)
        labels_MSE = np.zeros(9)
        labels_MSE[int(labels)]=1
        labels_MSE = torch.tensor(labels_MSE)
        labels_MSE = labels_MSE.float()
        #evaluate the MSE
        MSE_r_train = loss(outputs, labels_MSE)
        #evaluate the accuracy
        max_ind_out = torch.argmax(outputs) 
        #let's compute the backward propagation
        MSE_r_train.backward()
        #let's update the model parameters
        optimizer.step()

        MSE_r_train += MSE_r_train.item()
        if max_ind_out==labels:
            acc_r_train += 1
    
    t1 = time.time()
    print(t1-t0)
    
    MSE_train[epoch] = MSE_r_train/len(train)
    print('Epoch:', epoch+1, 'MSE:', MSE_train[epoch])
    acc_train[epoch] = 100*acc_r_train/len(train)
    print('Epoch:', epoch+1, 'accurancy:', acc_train[epoch],'%')
    
    #testing and accurancy
    for i in range(len(test)):
        inputs, labels = test[i]
        #evaluate the outputs
        outputs = NN_model(inputs)
        labels_MSE = np.zeros(9)
        labels_MSE[int(labels)]=1
        labels_MSE =  torch.tensor(labels_MSE)
        labels_MSE = labels_MSE.float()
        MSE_r_test = loss(outputs, labels_MSE)
        max_ind_out = torch.argmax(outputs) 
        
        #accumulate accurancy and loss
        MSE_r_test += MSE_r_test.item()
        if max_ind_out==labels:
            acc_r_test += 1
        
    MSE_test[epoch] = MSE_r_test/len(test)
    print('Epoch:', epoch+1, 'MSE test:', MSE_test[epoch])
    acc_test[epoch] = 100*acc_r_test/len(test)
    print('Epoch:', epoch+1, 'accurancy test:', acc_test[epoch],'%')

    if acc_train[epoch]>0.80:
        break
#save the model
torch.save(NN_model.state_dict(),'0502-678839043-Russo.zzz')

#plot
plt.figure(1)
plt.plot(np.arange(1,epoch_number+1),MSE_train,'b')
plt.plot(np.arange(1,epoch_number+1),MSE_test,'g')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.legend(['training set','test set'])
plt.title('MSE vs epoch')
plt.savefig('MSE vs epoch')

plt.figure(2)
plt.plot(np.arange(1,epoch_number+1),acc_train,'b')
plt.plot(np.arange(1,epoch_number+1),acc_test,'g')
plt.xlabel('epoch')
plt.ylabel('accurancy %')
plt.legend(['training set','test set'])
plt.title('accurancy vs epoch')
plt.savefig('accuracy vs epoch')

plt.show()