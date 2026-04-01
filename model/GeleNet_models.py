##################################################################################
##Our code is built on GeleNet. So in this code, GeleNet refers to our FreMaNet.##
##################################################################################
# MLChannelAttention3 refers to Mutual Assistance Channel Attention (MaCA).
import torch
import torch.nn as nn
import torch.nn.functional as F
# mobile_vit_xx_small mobile_vit_x_small mobile_vit_small
from model.MobileViT import mobile_vit_small
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax, Dropout

from typing import List, Callable
from torch import Tensor

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.PReLU(out_channel))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

# 深度可分离卷积 有relu
class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)

# 深度可分离卷积 无relu
class DSConv(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, bn=True, relu=False):
        super(DSConv, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, bn=bn, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)



# out = channel_shuffle(out, 2)
def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x

# 无relu
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


# MLChannelAttention3 refers to Mutual Assistance Channel Attention (MaCA).
class MLChannelAttention3(nn.Module):
    def __init__(self, channel, ratio=8):
        super(MLChannelAttention3, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.weight1 = nn.Sequential(
            nn.Conv2d(2*channel, channel // ratio, kernel_size=1),
            nn.BatchNorm2d(channel // ratio),
            nn.ReLU(True),
            nn.Conv2d(channel // ratio, channel, kernel_size=1),
            nn.Sigmoid())
        self.weight_1 = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, kernel_size=1),
            nn.BatchNorm2d(channel // ratio),
            nn.ReLU(True),
            nn.Conv2d(channel // ratio, channel, kernel_size=1),
            nn.Sigmoid())
        self.control1 = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

        self.weight2 = nn.Sequential(
            nn.Conv2d(2*channel, channel // ratio, kernel_size=1),
            nn.BatchNorm2d(channel // ratio),
            nn.ReLU(True),
            nn.Conv2d(channel // ratio, channel, kernel_size=1),
            nn.Sigmoid())
        self.weight_2 = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, kernel_size=1),
            nn.BatchNorm2d(channel // ratio),
            nn.ReLU(True),
            nn.Conv2d(channel // ratio, channel, kernel_size=1),
            nn.Sigmoid())
        self.control2 = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

        self.weight3 = nn.Sequential(
            nn.Conv2d(2*channel, channel // ratio, kernel_size=1),
            nn.BatchNorm2d(channel // ratio),
            nn.ReLU(True),
            nn.Conv2d(channel // ratio, channel, kernel_size=1),
            nn.Sigmoid())
        self.weight_3 = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, kernel_size=1),
            nn.BatchNorm2d(channel // ratio),
            nn.ReLU(True),
            nn.Conv2d(channel // ratio, channel, kernel_size=1),
            nn.Sigmoid())
        self.control3 = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)


    def forward(self, x3, x2, x1):
        x3_GAP = self.max_pool(x3)
        x2_GAP = self.max_pool(x2)
        x1_GAP = self.max_pool(x1)

        #self.weight3(torch.cat([x2_GAP, x1_GAP]), dim=1)是辅助权重，给辅助权重分配一个可以学习的参数torch.sigmoid(self.control3)
        #然后在与自身权重self.weight_3(x3_GAP)相加，用于调节x3
        ass_weight3 = self.weight3(torch.cat([x2_GAP, x1_GAP], dim=1))
        self_weight3 = self.weight_3(x3_GAP)
        x3 = x3 * ((1-torch.sigmoid(self.control3))*self_weight3 + torch.sigmoid(self.control3)*ass_weight3) + x3

        ass_weight2 = self.weight2(torch.cat([x3_GAP, x1_GAP], dim=1))
        self_weight2 = self.weight_2(x2_GAP)
        x2 = x2 * ((1-torch.sigmoid(self.control2))*self_weight2 + torch.sigmoid(self.control2)*ass_weight2) + x2

        ass_weight1 = self.weight1(torch.cat([x3_GAP, x2_GAP], dim=1))
        self_weight1 = self.weight_1(x1_GAP)
        x1 = x1 * ((1-torch.sigmoid(self.control1))*self_weight1 + torch.sigmoid(self.control1)*ass_weight1) + x1

        return x3, x2, x1

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class FreSA(nn.Module):
    def __init__(self, channel):
        super(FreSA, self).__init__()

        self.to_hidden = nn.Conv2d(channel, channel * 3, kernel_size=1)
        # device = torch.device("cpu")
        self.control = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
        # self.control_norm_imag = torch.sigmoid(nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True))

        self.weight = nn.Sequential(
            nn.Conv2d(channel, channel // 8, kernel_size=1),
            nn.BatchNorm2d(channel // 8),
            nn.ReLU(True),
            nn.Conv2d(channel // 8, channel, kernel_size=1),
            nn.Sigmoid())

        self.norm = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(True)
        # self.norm = LayerNorm(channel * 2, LayerNorm_type='WithBias')
        self.project_out = nn.Conv2d(channel, channel, kernel_size=1)


    def forward(self, x):
        q, k, v = self.to_hidden(x).chunk(3, dim=1)

        q_fft = torch.fft.fft2(q.float())
        k_fft = torch.fft.fft2(k.float())
        v_fft = torch.fft.fft2(v.float())

        # atten = q_fft * torch.conj(k_fft) * torch.sigmoid(self.control_norm)
        #atten = q_fft * k_fft * torch.sigmoid(self.control_norm)
        atten = q_fft * k_fft
        out = self.weight(atten.real) * v_fft

        out_ifft = self.relu(self.norm(torch.abs(torch.fft.ifft2(out))))

        output = self.project_out(torch.sigmoid(self.control)*out_ifft+x)
        #print(self.control)

        return output


class PDecoder(nn.Module):
    def __init__(self, channel):
        super(PDecoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = DSConv(channel, channel, stride=1)
        self.conv_upsample2 = DSConv(channel, channel, stride=1)
        self.conv_upsample3 = DSConv(channel, channel, stride=1)
        self.conv_upsample4 = DSConv(channel, channel, stride=1)
        self.conv_upsample5 = DSConv(2*channel, 2*channel, stride=1)

        self.conv_concat2 = DSConv(2*channel, 2*channel, stride=1)
        self.conv_concat3 = DSConv(3*channel, 3*channel, stride=1)
        self.conv4 = DSConv(3*channel, 3*channel, stride=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3): # x1: 32x22x22, x2: 32x44x44, x3: 32x88x88,
        x1_1 = x1 # 32x22x22
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2 # 32x44x44
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3 # 32x88x88

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1) # 32x44x44
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1) # 32x88x88
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x) # 1x88x88

        return x


class GeleNet(nn.Module):
    def __init__(self, channel=32):
        super(GeleNet, self).__init__()

        self.backbone = mobile_vit_small() # [64, 96, 128, 160]

        # self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './model/mobilevit_s.pt'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # input 3x352x352
        self.ChannelNormalization_1 = DSConv3x3(64, channel, stride=1)  # 64x88x88->32x88x88
        self.ChannelNormalization_2 = DSConv3x3(96, channel, stride=1) # 96x44x44->32x44x44
        self.ChannelNormalization_3 = DSConv3x3(128, channel, stride=1) # 128x22x22->32x22x22
        self.ChannelNormalization_4 = DSConv3x3(160, channel, stride=1) # 128x11x11->32x11x11

        # FreSA
        self.FreSA_1 = FreSA(channel)
        self.FreSA_2 = FreSA(channel)
        self.FreSA_3 = FreSA(channel)
        self.FreSA_4 = FreSA(channel)

        # MLCA refers to Mutual Assistance Channel Attention (MaCA).
        self.MLCA = MLChannelAttention3(channel)

        self.PDecoder = PDecoder(channel)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        # backbone
        MobileViT = self.backbone(x)
        x1 = MobileViT[1] # 64x88x88
        x2 = MobileViT[2] # 96x44x44
        x3 = MobileViT[3] # 128x22x22
        x4 = MobileViT[4] # 160x11x11

        x1_nor = self.ChannelNormalization_1(x1) # 32x88x88
        x2_nor = self.ChannelNormalization_2(x2) # 32x44x44
        x3_nor = self.ChannelNormalization_3(x3) # 32x22x22
        x4_nor = self.ChannelNormalization_4(x4) # 32x11x11

        x1_EN = self.FreSA_1(x1_nor)
        x2_EN = self.FreSA_2(x2_nor)
        x3_EN = self.FreSA_3(x3_nor)
        x4_EN = self.FreSA_4(x4_nor)

        # MLCA refers to Mutual Assistance Channel Attention (MaCA).
        x3_MLCA, x2_MLCA, x1_MLCA = self.MLCA(self.upsample_2(x4_EN)+x3_EN, self.upsample_2(x3_EN)+x2_EN, self.upsample_2(x2_EN)+x1_EN)

        prediction = self.upsample_4(self.PDecoder(x3_MLCA, x2_MLCA, x1_MLCA))


        return prediction


