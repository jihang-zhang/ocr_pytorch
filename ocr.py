import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax2d(nn.Module):
    def forward(self, input):
        assert input.dim() == 4, 'SpatialSoftmax2d requires a 4D tensor as input'
        
        N, C, H, W = input.size()
        input = input.view(N, C, -1).transpose(1, 2)
        return F.softmax(input, 1) # [N, HxW, K]


class RegionModule(nn.Module):
    def __init__(self):
        super(RegionModule, self).__init__()
        self.spatial_softmax = SpatialSoftmax2d()
    
    def forward(self, x, l):
        N, C, H, W = x.size()
        
        l = self.spatial_softmax(l)
        x = x.view(N, C, -1) # [N, C, HxW]
        return torch.bmm(x, l) # [N, C, K]


class ConvBnRelu1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ConvBnRelu1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class ContextModule(nn.Module):
    def __init__(self, in_channels, planes, activation='softmax'):
        super(ContextModule, self).__init__()
        self.planes = planes
        
        self.conv_psi = ConvBnRelu2d(in_channels, planes, kernel_size=1)
        self.conv_phi = ConvBnRelu1d(in_channels, planes, kernel_size=1)
        self.conv_delta = ConvBnRelu1d(in_channels, planes, kernel_size=1)
        self.conv_rho = ConvBnRelu2d(planes, in_channels, kernel_size=1)

        if activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Invalid activation type {}. Please choose one from ["softmax", "sigmoid"]'.format(activation))
        
    def forward(self, x, f):
        N, C, H, W = x.size()
        
        x = self.conv_psi(x) # [N, 256, H, W]
        f_phi = self.conv_phi(f) # [N, 256, K]
        w = torch.bmm(f_phi.transpose(1, 2), x.view(N, self.planes, -1)) # [N, K, HxW]
        w = self.activation(w) # [N, K, HxW]
        
        f_delta = self.conv_delta(f) # [N, 256, K]
        x = torch.bmm(f_delta, w).view(N, self.planes, H, W) # [N, 256, H, W]
        x = self.conv_rho(x) # [N, 512, H, W]
        
        return x


class OCR(nn.Module):
    def __init__(self, in_channels, planes, activation='softmax'):
        super(OCR, self).__init__()
        
        self.region = RegionModule()
        self.context = ContextModule(in_channels, planes, activation)
        self.conv_g = ConvBnRelu2d(in_channels*2, in_channels, 1)
    
    def forward(self, x, l):
        f = self.region(x, l)
        xo = self.context(x, f)
        
        out = torch.cat([x, xo], dim=1)
        out = self.conv_g(out)
        return out