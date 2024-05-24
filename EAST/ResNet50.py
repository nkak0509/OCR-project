import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels * 4)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IdentityBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet50(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet50, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64, 64, stride=1),
            IdentityBlock(256, 64),
            IdentityBlock(256, 64)
        )
        
        self.stage3 = nn.Sequential(
            ConvBlock(256, 128, stride=2),
            IdentityBlock(512, 128),
            IdentityBlock(512, 128),
            IdentityBlock(512, 128)
        )
        
        self.stage4 = nn.Sequential(
            ConvBlock(512, 256, stride=2),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256)
        )
        
        self.stage5 = nn.Sequential(
            ConvBlock(1024, 512, stride=2),
            IdentityBlock(2048, 512),
            IdentityBlock(2048, 512),
            IdentityBlock(2048, 512)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        f = []
        x = self.stage1(x)
        x = self.stage2(x)
        f.append(x)
        x = self.stage3(x)
        f.append(x)
        x = self.stage4(x)
        f.append(x)
        x = self.stage5(x)
        f.append(x)

        # x = self.avgpool(x)
        # x = self.flatten(x)
        # x = self.fc(x)
        
        return x, f

#resnet50 = ResNet50(num_classes=6)