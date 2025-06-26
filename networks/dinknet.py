import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchvision import models

nonlinearity = partial(F.relu, inplace=True)

# -------------------------------------------------------------------------
# Dilated centre blocks (original naming: dilate1…dilate4)
# -------------------------------------------------------------------------
class Dblock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dilate1 = nn.Conv2d(ch, ch, 3, padding=1, dilation=1)
        self.dilate2 = nn.Conv2d(ch, ch, 3, padding=2, dilation=2)
        self.dilate3 = nn.Conv2d(ch, ch, 3, padding=4, dilation=4)
        self.dilate4 = nn.Conv2d(ch, ch, 3, padding=8, dilation=8)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out1 = nonlinearity(self.dilate1(x))
        out2 = nonlinearity(self.dilate2(out1))
        out3 = nonlinearity(self.dilate3(out2))
        out4 = nonlinearity(self.dilate4(out3))
        return x + out1 + out2 + out3 + out4

class Dblock_more_dilate(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dilate1 = nn.Conv2d(ch, ch, 3, padding=1, dilation=1)
        self.dilate2 = nn.Conv2d(ch, ch, 3, padding=2, dilation=2)
        self.dilate3 = nn.Conv2d(ch, ch, 3, padding=4, dilation=4)
        self.dilate4 = nn.Conv2d(ch, ch, 3, padding=8, dilation=8)
        self.dilate5 = nn.Conv2d(ch, ch, 3, padding=16, dilation=16)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        o1 = nonlinearity(self.dilate1(x))
        o2 = nonlinearity(self.dilate2(o1))
        o3 = nonlinearity(self.dilate3(o2))
        o4 = nonlinearity(self.dilate4(o3))
        o5 = nonlinearity(self.dilate5(o4))
        return x + o1 + o2 + o3 + o4 + o5

# -------------------------------------------------------------------------
# Decoder block (original naming: deconv2)
# -------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1  = nn.Conv2d(in_ch, in_ch // 4, 1)
        self.norm1  = nn.BatchNorm2d(in_ch // 4)
        self.relu   = nonlinearity
        # original name deconv2 kept
        self.deconv2 = nn.ConvTranspose2d(in_ch // 4, in_ch // 4,
                                          3, stride=2, padding=1, output_padding=1)
        self.norm2  = nn.BatchNorm2d(in_ch // 4)
        self.conv3  = nn.Conv2d(in_ch // 4, out_ch, 1)
        self.norm3  = nn.BatchNorm2d(out_ch)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.deconv2(x)))
        x = self.relu(self.norm3(self.conv3(x)))
        return x

# -------------------------------------------------------------------------
class _Head(nn.Module):
    def _final(self, x):
        x = self.finaldeconv1(x)
        x = self.finalrelu1(x)
        x = self.finalconv2(x)
        x = self.finalrelu2(x)
        x = self.finalconv3(x)   # logits!
        return x

class DinkNet34(_Head):
    def __init__(self, num_classes=1, num_channels=3):
        super().__init__()
        filters = [64,128,256,512]
        resnet  = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        if num_channels!=3:
            self.firstconv = nn.Conv2d(num_channels,64,7,2,3,bias=False)
            self.firstconv.weight.data[:, :3] = resnet.conv1.weight.data.clone()
        else:
            self.firstconv = resnet.conv1
        self.firstbn=resnet.bn1
        self.firstrelu=resnet.relu
        self.firstmaxpool=resnet.maxpool
        self.encoder1=resnet.layer1
        self.encoder2=resnet.layer2
        self.encoder3=resnet.layer3
        self.encoder4=resnet.layer4

        self.dblock = Dblock(512)
        self.decoder4=DecoderBlock(filters[3],filters[2])
        self.decoder3=DecoderBlock(filters[2],filters[1])
        self.decoder2=DecoderBlock(filters[1],filters[0])
        self.decoder1=DecoderBlock(filters[0],filters[0])

        self.finaldeconv1=nn.ConvTranspose2d(filters[0],32,4,2,1)
        self.finalrelu1=nonlinearity
        self.finalconv2=nn.Conv2d(32,32,3,padding=1)
        self.finalrelu2=nonlinearity
        self.finalconv3=nn.Conv2d(32,num_classes,3,padding=1)

    def forward(self, x):
        x=self.firstrelu(self.firstbn(self.firstconv(x)))
        x=self.firstmaxpool(x)
        e1=self.encoder1(x)
        e2=self.encoder2(e1)
        e3=self.encoder3(e2)
        e4=self.encoder4(e3)
        e4=self.dblock(e4)
        d4=self.decoder4(e4)+e3
        d3=self.decoder3(d4)+e2
        d2=self.decoder2(d3)+e1
        d1=self.decoder1(d2)
        return self._final(d1)
