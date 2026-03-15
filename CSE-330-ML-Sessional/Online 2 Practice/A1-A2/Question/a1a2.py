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
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
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

# Number of classes
num_classes = len(dataset.classes)
print("Number of classes:", num_classes)

# ResNet Building Blocks
class ResidualBlock(nn.Module):
    """Basic Residual Block with skip connection"""
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        
        # TODO: Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        
        # TODO: Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(in_channels)
        )
        
    def forward(self, x):
        identity = x
        # TODO: Layer 1
        x = self.layer1(x)
        
        # TODO: Layer 2
        x = self.layer2(x)
        x += identity
        out = torch.relu(x)
        return out


class CustomResNet(nn.Module):
    """Custom ResNet architecture without nn.Sequential"""
    def __init__(self):
        super(CustomResNet, self).__init__()
        
        # TODO: Initial convolution layer
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=8,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3,2)
        )
        
        # TODO: Layer 1
        self.layer1 = ResidualBlock(64)

        # TODO: Layer 1 conv
        self.layer1_conv = nn.Sequential(
            nn.Conv2d(64,32,kernel_size=4,stride=1,padding=7),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # TODO: Layer 2
        self.layer2 = ResidualBlock(32)
        # TODO: Layer 2 conv
        self.layer2_conv = nn.Sequential(
            nn.Conv2d(32,16,kernel_size=5,stride=2,padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # TODO: Layer 3
        self.layer3 = nn.Sequential(
            ResidualBlock(16),
            nn.BatchNorm2d(16)
        )
        
        # TODO: Global average pooling and FC layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(16, 1)
    

    def forward(self, x):
        # TODO: Initial conv
        x = self.conv_layer(x)
        
        # TODO: Layer 1
        x = self.layer1(x)

        # TODO: Layer 1 conv
        x = self.layer1_conv(x)

        # TODO: Layer 2
        x = self.layer2(x)
        
        # TODO: Layer 2 conv
        x = self.layer2_conv(x)
    
        # TODO: Layer 3
        x = self.layer3(x)

        # TODO: Global pooling and FC layer
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)

        return x


# Initialize model
device = torch.device("cpu")

model = CustomResNet().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 4

print(f"\nTraining Custom ResNet with {sum(p.numel() for p in model.parameters())} parameters")
print(f"Device: {device}\n")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().to(device)

        # Forward pass
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        
        predicted = (torch.sigmoid(outputs) > 0.5).long()
        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
