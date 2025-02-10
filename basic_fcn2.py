import torch
import torch.nn as nn
import torchvision.models as models

class FCN2(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 1024, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3 = nn.BatchNorm2d(1024)

        self.leaky_relu = nn.LeakyReLU(inplace=True)
        
        self.deconv1 = nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, n_class, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)

    def forward(self, x):
        x = self.leaky_relu(self.bnd1(self.conv1(x)))
        x = self.leaky_relu(self.bnd2(self.conv2(x)))
        x = self.leaky_relu(self.bnd3(self.conv3(x)))

        x = self.leaky_relu(self.bn1(self.deconv1(x)))
        x = self.leaky_relu(self.bn2(self.deconv2(x)))
        x = self.deconv3(x)

        return x

class ResNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        
        # Load pre-trained ResNet34 and remove the fully connected and avgpool layers.
        resnet = models.resnet34(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        
        self.deconv5 = nn.ConvTranspose2d(32, n_class, kernel_size=3, stride=2,
                                          padding=1, output_padding=1)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Encoder: from (N, 3, 224, 224) to (N, 512, 7, 7)
        x = self.encoder(x)
        
        # Decoder: progressively upsample back to 224x224.
        x = self.relu(self.bn1(self.deconv1(x)))  # (N, 256, 14, 14)
        x = self.relu(self.bn2(self.deconv2(x)))    # (N, 128, 28, 28)
        x = self.relu(self.bn3(self.deconv3(x)))    # (N, 64, 56, 56)
        x = self.relu(self.bn4(self.deconv4(x)))    # (N, 32, 112, 112)
        x = self.deconv5(x)                         # (N, n_class, 224, 224)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNet, self).__init__()
        
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        
        d4 = self.down4(p3)
        p4 = self.pool4(d4)
        
        bottleneck = self.bottleneck(p4)
        
        up4 = self.up4(bottleneck)
        merge4 = torch.cat((d4, up4), dim=1)
        uc4 = self.up_conv4(merge4)
        
        up3 = self.up3(uc4)
        merge3 = torch.cat((d3, up3), dim=1)
        uc3 = self.up_conv3(merge3)
        
        up2 = self.up2(uc3)
        merge2 = torch.cat((d2, up2), dim=1)
        uc2 = self.up_conv2(merge2)
        
        up1 = self.up1(uc2)
        merge1 = torch.cat((d1, up1), dim=1)
        uc1 = self.up_conv1(merge1)
        
        output = self.final_conv(uc1)
        return output