import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # For CPU
    torch.cuda.manual_seed(seed)  # For GPU (if used)
    torch.cuda.manual_seed_all(seed)  # For all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False


set_seed(42)

# Image transformations
transform = transforms.Compose([
    # TODO: Resize to 224x224
    transforms.Resize((224,224)),
    # TODO: Convert to tensor
    transforms.ToTensor(),
    # TODO: Normalize with mean and std
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

# Load dataset
dataset = datasets.ImageFolder(
    root="images/",
    transform=transform
)

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=30,
    shuffle=True,
)

images, labels = next(iter(dataloader))

print("Batch image tensor shape:", images.shape)
print("Batch labels tensor shape:", labels.shape)

# Number of classes (auto-detected)
num_classes = len(dataset.classes)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        # TODO: Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )
        # TODO: Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=19,stride=1,padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )
        # TODO: Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=4,stride=2,padding=8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )
        # TODO: Block 4
        self.block4 = nn.Sequential(
            nn.Linear(64*16*16,128),
            nn.ReLU()
        )
        # TODO: Block 5
        self.block5 = nn.Sequential(
            nn.Linear(128,2)
        )

    def forward(self, x):
        # TODO: Block 1
        x = self.block1(x)
        # TODO: Block 2
        x = self.block2(x)
        # TODO: Block 3
        x = self.block3(x)
        x = x.view(x.size(0),-1)
        # TODO: Block 4
        x = self.block4(x)
        # TODO: Block 5
        x = self.block5(x)

        return x

class Adam:
    def __init__(self,params,lr,row1=0.9,row2=0.999,delta=1e-8):
        self.params = list(params)
        self.lr = lr
        self.row1 = row1
        self.row2 = row2
        self.delta = delta
        self.v = [torch.zeros_like(p.data) for p in self.params]
        self.r = [torch.zeros_like(p.data) for p in self.params]
        self.t = 0
    def step(self):
        self.t += 1
        for i,param in enumerate(self.params):
            if param.grad is None:
               continue
            self.v[i] = self.row1*self.v[i] + (1-self.row1)*param.grad
            self.r[i] = self.row2*self.r[i] + (1-self.row2)*param.grad*param.grad
            self.v[i] = self.v[i] / (1-self.row1**self.t)
            self.r[i] = self.r[i] / (1-self.row2**self.t)

            del_theta = -self.lr*(self.v[i]) /(torch.sqrt(self.r[i]) + self.delta)
            param.data += del_theta
    def zero_grad(self):
        for param in self.params:
          if param.grad is not None:
            param.grad.zero_() 
device = torch.device("cpu")

model = SimpleCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
