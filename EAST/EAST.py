import torch
import torch.nn as nn
import math 
input_size = 512 
from ResNet50 import ResNet50

class Concat(nn.Module):
    def __init__(self ,dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x1, x2):
        return torch.cat((x1, x2), self.d)
    
class conv_x_x(nn.Module):
    def __init__(self, in_channels =64, out_channels=64, kernel_size = 3):
        super(conv_x_x, self).__init__()
        if kernel_size == 1:
            self.conv = nn.Conv2d(in_channels= in_channels, out_channels=out_channels, kernel_size=kernel_size)
        else:
            self.conv = nn.Conv2d(in_channels= in_channels, out_channels=out_channels, kernel_size=kernel_size, padding= 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class EAST(nn.Module):
    def __init__(self, classes= 6):
        super(EAST, self).__init__()
        self.resnet = ResNet50(num_classes= classes)
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.concat = Concat()
        self.conv1 = conv_x_x(in_channels= 3072, out_channels= 128, kernel_size= 1)
        self.conv2 = conv_x_x(in_channels=128, out_channels=128, kernel_size=3)
        self.conv3 = conv_x_x(in_channels=640, out_channels=64, kernel_size=1)
        self.conv4 = conv_x_x(in_channels=64, out_channels=64, kernel_size=3)
        self.conv5 = conv_x_x(in_channels=320, out_channels=64, kernel_size=1)
        self.conv6 = conv_x_x(in_channels=64, out_channels=32, kernel_size=3)
        self.conv7 = conv_x_x(in_channels=32, out_channels=32, kernel_size=3)
        self.score = nn.Sequential(
            conv_x_x(in_channels=32, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
        self.geo_map = nn.Sequential(
            conv_x_x(in_channels=32, out_channels=4, kernel_size=1),
            nn.Sigmoid()
        )
        self.angle_map = nn.Sequential(
            conv_x_x(in_channels=32, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        _, f = self.resnet(x)

        #1
        h = f[3] # bs 2048 
        g = self.unpool(h) # bs 2048 
        #2
        c = self.conv1(self.concat(g, f[2])) # bs 3072(2048 + 1024) -> bs 128
        h = self.conv2(c) # bs 128
        g = self.unpool(h) # bs 128 
        #3
        c = self.conv3(self.concat(g, f[1])) # bs 640 (512 + 128) -> bs 64
        h = self.conv4(c) # bs 64
        g = self.unpool(h) # bs 64
        #4
        c = self.conv5(self.concat(g, f[0])) # b 320 (256 + 64) -> bs 64
        h = self.conv6(c) # bs 64 -> bs 32
        g = self.conv7(h) 

        ###################output####################
        score = self.score(g)

        geo_map = self.geo_map(g) * input_size

        angle_map = self.angle_map(g)
        angle_map = (angle_map - 0.5) * math.pi / 2

        geo = self.concat(geo_map, angle_map) 
        
        return score, geo
    

model = EAST()

input_tensor = torch.randn(1, 3, 512, 512)

# Pass the sample input through the block
score, geo = model(input_tensor)

# Print the shape of the output tensor
print("Output shape:", score.shape)
print("Output shape:", geo.shape)
