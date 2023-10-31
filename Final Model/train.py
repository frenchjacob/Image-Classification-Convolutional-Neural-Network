### this code trains the model 'model.pth'

from torch.utils.data import Subset
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torchvision.models import resnet18, ResNet18_Weights
from torch.optim import lr_scheduler
import os

# check if GPU is available
if torch.cuda.is_available():
  device = torch.device('cuda:0')
  print('GPU')

else:
  device = torch.device('cpu')
  print('CPU')

# Define the root directory of your dataset and your desired image size
train_root = 'traindata'

desired_size = (300, 300)  # set desired size for training data

# define transformation for the training set
train_transform = transforms.Compose([
    transforms.Resize(desired_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees = (-30,30))# Resize images to the desired size
])

# load & transform train data
train_dataset = ImageFolder(root=train_root, transform=train_transform)

#define batch size
batch_size = 64

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

classes = ('cherry','strawberry','tomato')

# set up resnet model
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs,3)

# move model to device
model.to(device)

# set loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)


for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print('Finished Training')

#save model
PATH = 'model.pth'
torch.save(model.state_dict(), PATH)