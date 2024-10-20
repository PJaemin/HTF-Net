import torch
import torch.nn as nn
import torch.nn.functional as F


class intensityTransform(nn.Module):
    def __init__(self, intensities, channels, **kwargs):
        super(intensityTransform, self).__init__(**kwargs)
        self.channels = channels
        self.scale = intensities - 1

    def get_config(self):
        config = super(intensityTransform, self).get_config()
        config.update({'channels': self.channels, 'scale': self.scale})
        return config

    def forward(self, inputs):
        images, transforms = inputs

        transforms = transforms.unsqueeze(3)  # Index tensor must have the same number of dimensions as input tensor

        images = 0.5 * images + 0.5
        images = torch.round(self.scale * images)
        images = images.type(torch.LongTensor)
        images = images.cuda()
        transforms = transforms.cuda()
        minimum_w = images.size(3)
        iter_n = 0
        temp = 1
        while minimum_w > temp:
            temp *= 2
            iter_n += 1

        for i in range(iter_n):
            transforms = torch.cat([transforms, transforms], dim=3)

        images = torch.split(images, 1, dim=1)
        transforms = torch.split(transforms, 1, dim=1)

        x = torch.gather(input=transforms[0], dim=2, index=images[0])
        y = torch.gather(input=transforms[1], dim=2, index=images[1])
        z = torch.gather(input=transforms[2], dim=2, index=images[2])

        xx = torch.cat([x, y, z], dim=1)

        return xx


class conv_block(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, strides, dropout_rate=0.1):
        super(conv_block, self).__init__()
        self.dropout_rate = dropout_rate
        padding = kernel_size//2
        self.cb_conv1 = nn.Conv2d(input_ch, output_ch, kernel_size, strides, padding=padding, bias=False)
        self.cb_batchNorm = nn.BatchNorm2d(output_ch)
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        x = self.cb_conv1(x)
        x = self.cb_batchNorm(x)
        x = self.swish(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)
        return x


class SFC_module(nn.Module):
    def __init__(self, in_ch, out_ch, expansion, num):
        super(SFC_module, self).__init__()
        exp_ch = int(in_ch * expansion)
        if num == 1:
            self.se_conv = nn.Conv2d(in_ch, exp_ch, 3, 1, 1, groups=in_ch)
        else:
            self.se_conv = nn.Conv2d(in_ch, exp_ch, 3, 2, 1, groups=in_ch)
        self.se_bn = nn.BatchNorm2d(exp_ch)
        self.se_relu = nn.ReLU()
        self.hd_conv = nn.Conv2d(exp_ch, exp_ch, 3, 1, 1, groups=in_ch)
        self.hd_bn = nn.BatchNorm2d(exp_ch)
        self.hd_relu = nn.ReLU()
        self.cp_conv = nn.Conv2d(exp_ch, out_ch, 1, 1, groups=in_ch)
        self.cp_bn = nn.BatchNorm2d(out_ch)
        self.pw_conv = nn.Conv2d(out_ch, out_ch, 1, 1)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.pw_relu = nn.ReLU()


    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        x = self.se_conv(x)
        x = self.se_bn(x)
        x = self.se_relu(x)
        x = self.hd_conv(x)
        x = self.hd_bn(x)
        x = self.hd_relu(x)
        x = self.cp_conv(x)
        x = self.cp_bn(x)
        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.pw_relu(x)
        return x


class HSFC_module(nn.Module):
    def __init__(self, in_ch, expansion):
        super(HSFC_module, self).__init__()
        exp_ch = int(in_ch * expansion)
        self.se_conv = nn.Conv1d(in_ch, exp_ch, 3, 1, 1, groups=in_ch)
        self.se_bn = nn.BatchNorm1d(exp_ch)
        self.se_relu = nn.ReLU()
        self.hd_conv = nn.Conv1d(exp_ch, exp_ch, 3, 1, 1, groups=in_ch)
        self.hd_bn = nn.BatchNorm1d(exp_ch)
        self.hd_relu = nn.ReLU()
        self.cp_conv = nn.Conv1d(exp_ch, in_ch, 1, 1, groups=in_ch)
        self.cp_bn = nn.BatchNorm1d(in_ch)
        self.pw_conv = nn.Conv1d(in_ch, in_ch, 1, 1)
        self.pw_bn = nn.BatchNorm1d(in_ch)
        self.pw_relu = nn.ReLU()

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        x = self.se_conv(x)
        x = self.se_bn(x)
        x = self.se_relu(x)
        x = self.hd_conv(x)
        x = self.hd_bn(x)
        x = self.hd_relu(x)
        x = self.cp_conv(x)
        x = self.cp_bn(x)
        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.pw_relu(x)
        return x


class Histogram_network(nn.Module):

    def __init__(self):
        super(Histogram_network, self).__init__()
        expansion = 4
        self.stage1 = HSFC_module(3, expansion)
        self.stage2 = HSFC_module(3, expansion)
        self.stage3 = HSFC_module(3, expansion)
        self.stage4 = HSFC_module(3, expansion)


    def forward(self, h):
        y = self.stage1(h)
        y = self.stage2(y)
        y = self.stage3(y)
        y = self.stage4(y)
        y = y.flatten(1)

        return y

class Attention_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Attention_block, self).__init__()
        self.g_conv = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=int(in_ch/2))
        self.g_bn = nn.BatchNorm2d(in_ch)
        self.g_relu = nn.ReLU()

        self.pw_conv = nn.Conv2d(in_ch, out_ch, 1, 1)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.pw_relu = nn.ReLU()

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, in1, in2):
        i1_1, i1_2, i1_3 = torch.chunk(in1, 3, dim=1)
        i2_1, i2_2, i2_3 = torch.chunk(in2, 3, dim=1)
        x = torch.cat([i1_1, i2_1, i1_2, i2_2, i1_3, i2_3], dim=1)

        x = self.g_conv(x)
        x = self.g_bn(x)
        x = self.g_relu(x)

        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.pw_relu(x)
        return x


class Image_network(nn.Module):

    def __init__(self):
        super(Image_network, self).__init__()
        expansion = 4
        C = 6

        self.histnet = Histogram_network()
        self.inputfus = Attention_block(6,6)

        self.stage2 = SFC_module(C, 2 * C, expansion, 1)
        self.stage3 = SFC_module(2 * C, 4 * C, expansion, 2)
        self.stage4 = SFC_module(4 * C, 8 * C, expansion, 3)
        self.stage5 = SFC_module(8 * C, 16 * C, expansion, 4)
        self.stage6 = SFC_module(16 * C, 32 * C, expansion, 5)
        self.stage7 = SFC_module(32 * C, 64 * C, expansion, 6)
        self.stage8 = SFC_module(64 * C, 128 * C, expansion, 7)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fusion_cv1 = nn.Conv2d(2, 2, 1)
        self.fusion_bn1 = nn.BatchNorm2d(2)
        self.fusion_ru1 = nn.ReLU()
        self.fusion_cv2 = nn.Conv2d(2, 1, 1)
        self.fusion_bn2 = nn.BatchNorm2d(1)
        self.fusion_ru2 = nn.ReLU()
        self.fusion_FC = nn.Linear(768, 768)
        self.fusion_sig = nn.Sigmoid()
        self.FC1 = nn.Linear(768, 768)
        self.FC2 = nn.Linear(768, 768)
        self.FC3 = nn.Linear(768, 768)

        self.intensity_trans = intensityTransform(intensities=256, channels=3)

    def forward(self, x, att, hist):

        x_256 = F.interpolate(x, 256)
        y = self.inputfus(x_256, att)

        y = self.stage2(y)
        y = self.stage3(y)
        y = self.stage4(y)
        y = self.stage5(y)
        y = self.stage6(y)
        y = self.stage7(y)
        y = self.stage8(y)
        y = self.gap(y)
        y = y.squeeze(2)
        y = y.squeeze(2)

        h = self.histnet(hist)
        h = h.unsqueeze(1)
        h = h.unsqueeze(3)
        ya = y.unsqueeze(1)
        ya = ya.unsqueeze(3)
        ya = torch.cat([ya, h], dim=1)

        ya = self.fusion_cv1(ya)
        ya = self.fusion_bn1(ya)
        ya = self.fusion_ru1(ya)
        ya = self.fusion_cv2(ya)
        ya = self.fusion_bn2(ya)
        ya = self.fusion_ru2(ya)
        ya = ya.squeeze(3)
        ya = ya.squeeze(1)
        ya = self.fusion_FC(ya)
        ya = self.fusion_sig(ya)
        y = y * ya + y
        y = self.FC1(y)
        y = self.FC2(y)
        y = self.FC3(y)

        y = y.unsqueeze(1)
        y = torch.chunk(y, 3, dim=2)
        tf = torch.cat(y, dim=1)
        tf = torch.tanh(tf)
        xy = self.intensity_trans((x, tf))

        return xy, tf

