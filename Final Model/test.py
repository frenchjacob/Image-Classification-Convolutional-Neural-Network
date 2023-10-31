### this code loads 'model.pth' and tests it on the test set
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torchvision.models import resnet18, ResNet18_Weights

# check if GPU is available
if torch.cuda.is_available():
  device = torch.device('cuda:0')
  print('GPU')

else:
  device = torch.device('cpu')
  print('CPU')

# initialise model
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs,3)

# load trained model
PATH = 'model.pth'
model = model
model.load_state_dict(torch.load(PATH))

model.to(device)

test_root = 'testdata'

desired_size = (300,300)

test_transform = transforms.Compose([
    transforms.Resize(desired_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))# Resize images to the desired size
])

test_dataset = ImageFolder(root=test_root, transform=test_transform)

batch_size = 64

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('cherry','strawberry','tomato')


model.eval()


correct = 0
total = 0
# test model on the test set
with torch.no_grad():
    for data in testloader:
        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)


        # calculate outputs by running images through the network
        outputs = model(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the CNN: {100 * correct // total} %')



