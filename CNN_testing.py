import torch
import torch.nn as nn
from torchvision import transforms
import os
import cv2

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


#define all the possible classes
classes = ['Circle', 'Square', 'Octagon','Heptagon', 'Nonagon', 'Star', 'Hexagon', 'Pentagon', 'Triangle']

#list all the files in the directory ending with .png
directory = '.'  # Setting to the current directory
# Get a list of all files in the directory
files = os.listdir(directory)
# Filter the list to only include .png files
file_names = [file for file in files if file.endswith('.png')]

#inference model
model = CNN()
model.load_state_dict(torch.load('0502-678839043-Russo.zzz'))
print(file_names)
for names in file_names:
    image = cv2.imread(names)
    image = cv2.resize(image, [40,40], interpolation = cv2.INTER_AREA)
    image = (image/127.5)-1.0
    transform = transforms.ToTensor()
    image = transform(image)
    image = image.float()
    outputs = model(image)
    max_ind_out = torch.argmax(outputs)
    print(names,':',classes[max_ind_out])