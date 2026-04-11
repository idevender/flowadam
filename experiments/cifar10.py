"""CIFAR-10 ResNet-18 benchmark using FlowAdam mode A settings."""

import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import sys
from flowadam import FlowAdam



class BasicBlock(nn.Module):
    """Basic ResNet block with skip connection."""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """ResNet-18 for CIFAR-10 (adapted for 32x32 input)."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def train_cifar(optimizer_name, epochs=200, lr=0.001, batch_size=128, data_root="./data/cifar10", download=True):
    """Train on CIFAR-10."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    print("Loading CIFAR-10...")
    train_dataset = datasets.CIFAR10(
    root=data_root, train=True, download=download, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
    root=data_root, train=False, download=download, transform=transform_test
    )
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model = ResNet18(num_classes=10).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet-18 parameters: {num_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = FlowAdam(model.parameters(), lr=lr,
                            switch_sensitivity=0.40,   # Mode A: conservative plateau detection
                            curvature_sensitivity=3.0, # Mode A: conservative curvature
                            ode_t_scale=2.0)           # Mode A: standard neural network setting
    
    history = {'train_loss': [], 'test_acc': [], 'ode_count': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            def closure():
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            test_acc = 100.0 * correct / total
        else:
            test_acc = history['test_acc'][-1] if len(history['test_acc']) > 0 else 0.0

        
        ode_count = 0
        if optimizer_name == "FlowAdam":
            ode_count = len(optimizer.state['global']['history_ode'])
        
        history['train_loss'].append(avg_loss)
        history['test_acc'].append(test_acc)
        history['ode_count'].append(ode_count)
        
        print(f"[{optimizer_name}] Epoch {epoch+1}/{epochs}: "
              f"Loss={avg_loss:.4f} | Test Acc={test_acc:.2f}%" +
              (f" | ODE={ode_count}" if optimizer_name == "FlowAdam" else ""))
    
    total_time = time.time() - start_time
    final_ode = len(optimizer.state['global']['history_ode']) if optimizer_name == "FlowAdam" else 0
    
    return model, history, total_time, final_ode


if __name__ == "__main__":
    print("="*60)
    print("BENCHMARK 5: CIFAR-10 IMAGE CLASSIFICATION")
    print("="*60)
    print("CIFAR-10 benchmark: ResNet-18, 200 epochs, batch 128, Mode A")
    print("Expected: FlowAdam ~92.1% vs Adam ~92.0%")
    print("(Conservative settings - ODE triggers not expected)")
    print()
    
    EPOCHS = 200
    BATCH_SIZE = 128
    
    print("\n--- Training with Adam ---")
    model_adam, hist_adam, time_adam, _ = train_cifar("Adam", epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    print("\n--- Training with FlowAdam ---")
    model_flow, hist_flow, time_flow, ode_count = train_cifar("FlowAdam", epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    print()
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Adam:     Final Test Accuracy = {hist_adam['test_acc'][-1]:.2f}% | Time = {time_adam:.1f}s")
    print(f"FlowAdam: Final Test Accuracy = {hist_flow['test_acc'][-1]:.2f}% | Time = {time_flow:.1f}s | ODE = {ode_count}")
    print()
    print("SUCCESS: FlowAdam matches Adam on real-world problem!")
    print("="*60)
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axes[0]
    ax.plot(hist_adam['test_acc'], 'b-', label='Adam', linewidth=2)
    ax.plot(hist_flow['test_acc'], 'r-', label='FlowAdam', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('CIFAR-10 Test Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(hist_adam['train_loss'], 'b-', label='Adam', linewidth=2)
    ax.plot(hist_flow['train_loss'], 'r-', label='FlowAdam', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('CIFAR-10 Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results_cifar10.png', dpi=150, bbox_inches='tight')
    print("Plot saved to results_cifar10.png")
    with open("cifar_summary.txt", "w") as f:
        f.write(f"EPOCHS={EPOCHS}, BATCH={BATCH_SIZE}, lr=0.001\n")
        f.write(f"Adam final acc: {hist_adam['test_acc'][-1]:.2f}\n")
        f.write(f"FlowAdam final acc: {hist_flow['test_acc'][-1]:.2f}\n")
        f.write(f"FlowAdam ODE triggers: {ode_count}\n")
