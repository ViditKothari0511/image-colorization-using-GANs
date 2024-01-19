import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms, utils
from PIL import Image


class CDk(nn.Module):
    def __init__(self, in_features, out_features, is_leaky=False):
        super(CDk, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv2d(in_features, out_features, kernel_size=3, padding=1))
        self.layers.append(nn.InstanceNorm2d(out_features))
        self.layers.append(nn.Dropout(0.5))
        if is_leaky:
            self.layers.append(nn.LeakyReLU(0.2))
        else:
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(out_features, out_features, 3, padding=1)),
        self.layers.append(nn.InstanceNorm2d(out_features))
        self.layers.append(nn.Dropout(0.5))
        if is_leaky:
            self.layers.append(nn.LeakyReLU(0.2))
        else:
            self.layers.append(nn.ReLU())

    def forward(self, x):
        for each_layer in self.layers:
            x = each_layer(x)
        return x


class Ck(nn.Module):
    def __init__(self, in_features, out_features, is_leaky=False, batch_norm=True):
        super(Ck, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_features, out_features, kernel_size=3, padding=1))
        if batch_norm:
            self.layers.append(nn.InstanceNorm2d(out_features))
        if is_leaky:
            self.layers.append(nn.LeakyReLU(0.2))
        else:
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(out_features, out_features, 3, padding=1))
        if batch_norm:
            self.layers.append(nn.InstanceNorm2d(out_features))
        if is_leaky:
            self.layers.append(nn.LeakyReLU(0.2))
        else:
            self.layers.append(nn.ReLU())

    def forward(self, x):
        for each_layer in self.layers:
            x = each_layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_features, out_features):
        super(Upsample, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.ConvTranspose2d(in_features, out_features, 3, 2, 1, 1))
        self.layers.append(nn.InstanceNorm2d(out_features))
        self.layers.append(nn.ReLU())

    def forward(self, x):
        for each_layer in self.layers:
            x = each_layer(x)
        return x


class Generator(nn.Module):
    # encoder_plan = ['C64', 'C128', 'C256', 'C512', 'C512']
    # decoder_plan = ['CD512', 'CD1024', 'CD512', 'CD256', 'CD128']
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.ModuleDict()
        in_features = 3
        self.layers['pool'] = nn.MaxPool2d(2)
        self.layers['en1'] = Ck(in_features, 64, True, False)
        self.layers['en2'] = Ck(64, 128, True)
        self.layers['en3'] = Ck(128, 256, True)
        self.layers['en4'] = Ck(256, 512, True)
        self.layers['en5'] = Ck(512, 1024, True)

        self.layers['up1'] = Upsample(1024, 512)
        self.layers['up2'] = Upsample(512, 256)
        self.layers['up3'] = Upsample(256, 128)
        self.layers['up4'] = Upsample(128, 64)

        self.layers['dec1'] = CDk(1024, 512, is_leaky=False)
        self.layers['dec2'] = CDk(512, 256, is_leaky=False)
        self.layers['dec3'] = Ck(256, 128, is_leaky=False)
        self.layers['dec4'] = Ck(128, 64, is_leaky=False)
        self.layers['final'] = nn.Conv2d(64, 3, kernel_size=1, padding=1, stride=1)
        self.layers['tan'] = nn.Sigmoid()

    def forward(self, x):
        x1 = self.layers['en1'](x)
        x2 = self.layers['en2'](self.layers['pool'](x1))
        x3 = self.layers['en3'](self.layers['pool'](x2))
        x4 = self.layers['en4'](self.layers['pool'](x3))
        x5 = self.layers['en5'](self.layers['pool'](x4))

        x = self.layers['up1'](x5)
        x = torch.cat((x, x4), dim=1)
        x = self.layers['dec1'](x)
        x = self.layers['up2'](x)
        x = torch.cat((x, x3), dim=1)
        x = self.layers['dec2'](x)
        x = self.layers['up3'](x)
        x = torch.cat((x, x2), dim=1)
        x = self.layers['dec3'](x)
        x = self.layers['up4'](x)
        x = torch.cat((x, x1), dim=1)
        x = self.layers['dec4'](x)
        x = self.layers['final'](x)
        x = self.layers['tan'](x)
        return x


class Discriminator(nn.Module):
    plan = ['C64', 'C128', 'C256', 'C512']

    def __init__(self):
        super(Discriminator, self).__init__()
        in_features = 3

        self.conv_color = Ck(in_features, 64, batch_norm=False)
        self.conv_mono = Ck(3, 64, batch_norm=False)

        self.conv2 = Ck(64, 128)
        self.conv3 = Ck(128, 256)
        self.conv4 = Ck(256, 512)

        self.pooling = nn.MaxPool2d(2)
        self.conv6 = nn.Conv2d(512, 1, kernel_size=1, padding=1, stride=1)
        self.final = nn.Linear(4356, 1)

    def forward(self, x_color, x_mono):
        out_color = self.conv_color(x_color)
        out_color = self.pooling(out_color)

        out_mono = self.conv_mono(x_mono)
        out_mono = self.pooling(out_mono)

        out_color = self.conv2(out_color)
        out_mono = self.conv2(out_mono)

        out = torch.cat((out_color, out_mono), 1)
        out = self.pooling(out)

        out = self.conv4(out)
        out = self.pooling(out)

        out = self.conv6(out)
        out = out.view(out.size(0), -1)
        out = self.final(out)

        activation = nn.Sigmoid()
        out = activation(out)

        return out
    