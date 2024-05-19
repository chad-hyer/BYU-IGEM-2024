import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb
wandb.login()
config={"num_classes":10,"learning_rate":0.001,"batch_size":128,"num_epochs":10}
wandb.init(project="myfirstcnn", config=config)

class myfirstcnn(nn.Module):
    def __init__(self,num_classes):
        super(myfirstcnn, self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu=nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32, kernel_size=3, stride=1, padding=1)
        self.flatten=nn.Flatten()
        self.fc1=nn.Linear(32*8*8,64)
        self.dropout=nn.Dropout(0.5)
        self.fc2= nn.Linear(64,num_classes)

    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)
        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool(x)
        x=self.flatten(x)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return x
num_classes=10
learning_rate=0.001
batch_size=128
num_epochs=10

model=myfirstcnn(config['num_classes'])

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=config['learning_rate'])

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5,),(0.5,0.5,0.5))])
train_dataset = datasets.CIFAR10(root='./data',train=True,transform=transform,download=True)
test_dataset= datasets.CIFAR10(root='./data',train=False,transform=transform)
train_subset=torch.utils.data.Subset(train_dataset, range(500))
test_subset=torch.utils.data.Subset(test_dataset, range(100))
train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

for epoch in range(config['num_epochs']):
    running_loss = 0.0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

correct=0
total=0
with torch.no_grad():
    for images, labels in test_loader:
        outputs= model(images)
        _, predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
print(f"Accuracy on test set: {100 * correct / total}%")
wandb.log({"train/train_loss":loss})
