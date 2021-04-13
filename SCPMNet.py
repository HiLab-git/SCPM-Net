import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def Conv_Block(in_planes, out_planes, stride=1):
    """3x3x3 convolution with batchnorm and relu"""
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False),
                         nn.BatchNorm3d(out_planes),
                         nn.ReLU(inplace=True))


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Norm(nn.Module):
    def __init__(self, N):
        super(Norm, self).__init__()
        self.normal = nn.BatchNorm3d(N)

    def forward(self, x):
        return self.normal(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = Norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = Norm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = Norm(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = Norm(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = Norm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SAC(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(SAC, self).__init__()

        self.conv_1 = nn.Conv3d(
            input_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv3d(
            input_channel, out_channel, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv_5 = nn.Conv3d(
            input_channel, out_channel, kernel_size=3, stride=1, padding=3, dilation=3)
        self.weights = nn.Parameter(torch.ones(3))
        self.softmax = nn.Softmax(0)

    def forward(self, inputs):
        feat_1 = self.conv_1(inputs)
        feat_3 = self.conv_3(inputs)
        feat_5 = self.conv_5(inputs)
        weights = self.softmax(self.weights)
        feat = feat_1 * weights[0] + feat_3 * weights[1] + feat_5 * weights[2]
        return feat


class Pyramid_3D(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=256, using_sac=False):
        super(Pyramid_3D, self).__init__()

        self.P5_1 = nn.Conv3d(C5_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv3d(feature_size, feature_size, kernel_size=3, stride=1,
                              padding=1) if not using_sac else SAC(feature_size, feature_size)

        self.P4_1 = nn.Conv3d(C4_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv3d(feature_size, feature_size, kernel_size=3, stride=1,
                              padding=1) if not using_sac else SAC(feature_size, feature_size)

        self.P3_1 = nn.Conv3d(C3_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv3d(feature_size, feature_size, kernel_size=3, stride=1,
                              padding=1) if not using_sac else SAC(feature_size, feature_size)

        self.P2_1 = nn.Conv3d(C2_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv3d(feature_size, feature_size, kernel_size=3, stride=1,
                              padding=1) if not using_sac else SAC(feature_size, feature_size)

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x
        P2_x = self.P2_2(P2_x)

        return [P2_x, P3_x, P4_x, P5_x]


class Attention_SE_CA(nn.Module):
    def __init__(self, channel):
        super(Attention_SE_CA, self).__init__()
        self.Global_Pool = nn.AdaptiveAvgPool3d(1)
        self.FC1 = nn.Sequential(nn.Linear(channel, channel),
                                 nn.ReLU(), )
        self.FC2 = nn.Sequential(nn.Linear(channel, channel),
                                 nn.Sigmoid(), )

    def forward(self, x):
        G = self.Global_Pool(x)
        G = G.view(G.size(0), -1)
        fc1 = self.FC1(G)
        fc2 = self.FC2(fc1)
        fc2 = torch.unsqueeze(fc2, 2)
        fc2 = torch.unsqueeze(fc2, 3)
        fc2 = torch.unsqueeze(fc2, 4)
        return fc2*x


class SCPMNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 32
        super(SCPMNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = Norm(32)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = Norm(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.atttion1 = Attention_SE_CA(32)
        self.atttion2 = Attention_SE_CA(32)
        self.atttion3 = Attention_SE_CA(64)
        self.atttion4 = Attention_SE_CA(64)
        self.conv_1 = Conv_Block(64 + 3, 64)
        self.conv_2 = Conv_Block(64 + 3, 64)
        self.conv_3 = Conv_Block(64 + 3, 64)
        self.conv_4 = Conv_Block(64 + 3, 64)
        self.conv_8x = Conv_Block(64, 64)
        self.conv_4x = Conv_Block(64, 64)
        self.conv_2x = Conv_Block(64, 64)
        self.convc = nn.Conv3d(64, 1, kernel_size=1, stride=1)
        self.convr = nn.Conv3d(64, 1, kernel_size=1, stride=1)
        self.convo = nn.Conv3d(64, 3, kernel_size=1, stride=1)
        if block == BasicBlock:
            fpn_sizes = [self.layer1[layers[0]-1].conv2.out_channels, self.layer2[layers[1]-1].conv2.out_channels,
                         self.layer3[layers[2]-1].conv2.out_channels, self.layer4[layers[3]-1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer1[layers[0]-1].conv3.out_channels, self.layer2[layers[1]-1].conv3.out_channels,
                         self.layer3[layers[2]-1].conv3.out_channels, self.layer4[layers[3]-1].conv3.out_channels]

        self.fpn = Pyramid_3D(
            fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3], feature_size=64)  # 256

        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                Norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, c_2, c_4, c_8, c_16):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.atttion1(x)
        x1 = self.layer1(x)
        x1 = self.atttion2(x1)
        x2 = self.layer2(x1)
        x2 = self.atttion3(x2)
        x3 = self.layer3(x2)
        x3 = self.atttion4(x3)
        x4 = self.layer4(x3)
        feats = self.fpn([x1, x2, x3, x4])
        feats[0] = torch.cat([feats[0], c_2], 1)
        feats[0] = self.conv_1(feats[0])
        feats[1] = torch.cat([feats[1], c_4], 1)
        feats[1] = self.conv_2(feats[1])
        feats[2] = torch.cat([feats[2], c_8], 1)
        feats[2] = self.conv_3(feats[2])
        feats[3] = torch.cat([feats[3], c_16], 1)
        feats[3] = self.conv_4(feats[3])

        feat_8x = F.upsample(
            feats[3], scale_factor=2, mode='nearest') + feats[2]
        feat_8x = self.conv_8x(feat_8x)
        feat_4x = F.upsample(
            feat_8x, scale_factor=2, mode='nearest') + feats[1]
        feat_4x = self.conv_4x(feat_4x)
        feat_2x = F.upsample(feat_4x, scale_factor=2, mode='nearest')
        feat_2x = self.conv_2x(feat_2x)
        Cls1 = self.convc(feats[0])
        Cls2 = self.convc(feat_2x)
        Reg1 = self.convr(feats[0])
        Reg2 = self.convr(feat_2x)
        Off1 = self.convo(feats[0])
        Off2 = self.convo(feat_2x)
        output = {}
        output['Cls1'] = Cls1
        output['Reg1'] = Reg1
        output['Off1'] = Off1
        output['Cls2'] = Cls2
        output['Reg2'] = Reg2
        output['Off2'] = Off2
        return output


def scpmnet18(**kwargs):
    """Using ResNet-18 as backbone for SCPMNet.
    """
    model = SCPMNet(BasicBlock, [2, 2, 3, 3], **kwargs)  # [2,2,2,2]
    return model


if __name__ == '__main__':
    device = torch.device("cuda")
    input = torch.ones(1, 1, 96, 96, 96).to(device)
    coord_2 = torch.ones(1, 3, 48, 48, 48).to(device)
    coord_4 = torch.ones(1, 3, 24, 24, 24).to(device)
    coord_8 = torch.ones(1, 3, 12, 12, 12).to(device)
    coord_16 = torch.ones(1, 3, 6, 6, 6).to(device)
    label = torch.ones(1, 3, 5).to(device)
    net = scpmnet18().to(device)
    net.eval()
    out = net(input, coord_2, coord_4, coord_8, coord_16)
    print(out)
    print('finish')
