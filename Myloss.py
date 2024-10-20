import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import torchvision

import numpy as np
from PIL import Image


class perceptual_loss(nn.Module):
    def __init__(self, requires_grad=False):
        super(perceptual_loss, self).__init__()

        self.maeloss = torch.nn.L1Loss()
        vgg = vgg16(pretrained=True).cuda()

        vgg_pretrained_features = vgg.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 4):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 6):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, X, Y):
        # print(X.shape)
        xx = self.slice1(X)
        fx2 = xx
        xx = self.slice2(xx)
        fx4 = xx
        xx = self.slice3(xx)
        fx6 = xx

        yy = self.slice1(Y)
        fy2 = yy
        yy = self.slice2(yy)
        fy4 = yy
        yy = self.slice3(yy)
        fy6 = yy

        loss_p = self.maeloss(fx2, fy2) + self.maeloss(fx4, fy4) + self.maeloss(fx6, fy6)

        return loss_p


class monotonous_loss(nn.Module):
    def __init__(self, requires_grad=False):
        super(monotonous_loss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        # x = x * 0.5 + 0.5
        # x = torch.round(255 * x)

        g_sum = torch.zeros(batch_size, 3).cuda()

        for i in range(0, 255):
            g = x[:, :, i + 1] - x[:, :, i]
            h = g.clone()
            g[g < 0] = 1.
            g[h >= 0] = 0.
            g_sum += g
        g_sum = g_sum/255

        loss_m = torch.mean(g_sum)

        return loss_m


class attention_loss(nn.Module):
    def __init__(self):
        super(attention_loss, self).__init__()
        self.mseloss = nn.L1Loss().cuda()

    def forward(self, att, att_gt):
        att_gt = F.interpolate(att_gt, 256, mode='bilinear', align_corners=True)

        return self.mseloss(att, att_gt)

class transfunction_loss(nn.Module):
    def __init__(self):
        super(transfunction_loss, self).__init__()
        self.maeloss = nn.L1Loss().cuda()

    def forward(self, tf, htf, gt):

        return 0.6*self.maeloss(tf, gt) + 0.4*self.maeloss(htf, gt)