import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder [N, 1, 150, 150]
        self.e_conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1) # N, 8, 75, 75
        self.e_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)# N, 16, 38, 38
        self.e_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)# N, 32, 19, 19
        self.e_conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)# N, 64, 10, 10
        self.e_conv5 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=9, stride=2) # N, 128, 1, 1
        self.e_bc1 = nn.BatchNorm2d(32)
        self.e_bc2 = nn.BatchNorm2d(64)
        self.e_bc3 = nn.BatchNorm2d(128)
        
        #  Decoder [N, 128, 1, 1]
        self.d_conv1 = nn.ConvTranspose2d(512, 128, 9, stride=2, padding=0, output_padding=1) # N, 64 , 10, 10 
        self.d_conv2 = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=0) # N, 32 , 19, 19
        self.d_conv3 = nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, output_padding=1) # N, 16 , 38, 38
        self.d_conv4 = nn.ConvTranspose2d(64, 16, 3, stride=2, padding=1, output_padding=0) # N, 8, 75, 75
        self.d_conv5 = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1) # N, 3, 150, 150
        self.d_out = nn.ConvTranspose2d(4, 3, 1, stride=1, padding=0, output_padding=0) # N, 3, 150, 150
    
    def forward(self, x):
        x_ec1 = F.leaky_relu(self.e_conv1(x), 0.1)
        x_ec2 = self.e_bc1(F.leaky_relu(self.e_conv2(x_ec1), 0.1))
        x_ec3 = self.e_bc2(F.leaky_relu(self.e_conv3(x_ec2), 0.1))
        x_ec4 = self.e_bc3(F.leaky_relu(self.e_conv4(x_ec3), 0.1))
        x_ec5 = F.leaky_relu(self.e_conv5(x_ec4), 0.1)
        
        x_dc1 = F.leaky_relu(self.d_conv1(x_ec5), 0.1)
        x_dc1 = torch.cat((x_dc1, x_ec4), dim=1)
        x_dc2 = F.leaky_relu(self.d_conv2(x_dc1), 0.1)
        x_dc2 = torch.cat((x_dc2, x_ec3), dim=1)
        x_dc3 = F.leaky_relu(self.d_conv3(x_dc2), 0.1)
        x_dc3 = torch.cat((x_dc3, x_ec2), dim=1)
        x_dc4 = F.leaky_relu(self.d_conv4(x_dc3), 0.1)
        x_dc4 = torch.cat((x_dc4, x_ec1), dim=1)
        x_dc5 = F.leaky_relu(self.d_conv5(x_dc4), 0.1)
        x_dc5 = torch.cat((x_dc5, x), dim=1)
        x_out = self.d_out(x_dc5)
        
        return x_out
