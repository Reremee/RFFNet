from torch import nn
import torch
from .tensor_ops import cus_sample
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np
import scipy.stats as st


class BasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)

class MRB_1(nn.Module):
    def __init__(self, mid_c):
        super(MRB_1, self).__init__()
        self.s2m = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2m = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.bnm = nn.BatchNorm2d(mid_c)
        self.relu = nn.ReLU(True)

    def forward(self, in_m, in_s):
        # stage 1
        if in_s.size()!=in_m.size():
            in_s = F.interpolate(in_s, size=in_m.size()[2:], mode="bilinear", align_corners=False)
            
        s2m = self.s2m(in_s)
        m2m = self.m2m(in_m)
        
        out = self.relu(self.bnm(s2m + m2m) + in_m)
        return out
    
class MRB_2(nn.Module):
    def __init__(self, mid_c):
        super(MRB_2, self).__init__()
        self.s12m = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.s22m = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2m = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.bnm = nn.BatchNorm2d(mid_c)
        self.relu = nn.ReLU(True)

    def forward(self, in_m, in_s1, in_s2):
        # stage 1
        if in_s1.size()!=in_m.size():
            in_s1 = F.interpolate(in_s1, size=in_m.size()[2:], mode="bilinear", align_corners=False)
        if in_s2.size()!=in_m.size():
            in_s2 = F.interpolate(in_s2, size=in_m.size()[2:], mode="bilinear", align_corners=False)
        s12m = self.s12m(in_s1)
        s22m = self.s22m(in_s2)
        m2m = self.m2m(in_m)
        
        out = self.relu(self.bnm(s12m + s22m + m2m) + in_m)
        return out

class AGG(nn.Module):
    def __init__(self, mid_c):
        super(AGG, self).__init__()
        self.conv2 = MRB_2(mid_c)
        self.conv3 = MRB_2(mid_c)
        self.conv4 = MRB_2(mid_c)

    def forward(self, *xs):
        # in_data_2, in_data_4, in_data_8, in_data_16, in_data_32
        out_xs = []
        out_xs.append(self.conv2(xs[1], xs[0], xs[2]))
        out_xs.append(self.conv3(xs[2], xs[1], xs[3]))
        out_xs.append(self.conv4(xs[3], xs[2], xs[4]))
        return out_xs

class FRM_l(nn.Module):
    def __init__(self, mid_c = 64):
        super(FRM_l, self).__init__()
        self.tom_1 = MRB_1(mid_c)
        self.tod_1 = MRB_1(mid_c)
        self.tom_2 = MRB_1(mid_c)
        self.tod_2 = MRB_1(mid_c)
        self.relu = nn.ReLU(True)
    def forward(self, in_h, in_m, in_dh, in_d):
        m_1 = self.tom_1(in_m, in_h)
        d_1 = self.tod_1(in_d, in_dh)
        m_2 = self.tom_2(m_1, d_1)
        d_2 = self.tod_2(d_1, m_1)
        return m_2, d_2
    
class FRM_h(nn.Module):
    def __init__(self, mid_c = 64):
        super(FRM_h, self).__init__()
        self.tom_1 = MRB_1(mid_c)
        self.tod_1 = MRB_1(mid_c)
        self.tom_2 = MRB_1(mid_c)
        self.tod_2 = MRB_1(mid_c)
        self.relu = nn.ReLU(True)
    def forward(self, in_m, in_l, in_d, in_dl):
        m_1 = self.tom_1(in_m, in_l)
        d_1 = self.tod_1(in_d, in_dl)
        m_2 = self.tom_2(m_1, d_1)
        d_2 = self.tod_2(d_1, m_1)
        return m_2, d_2

class FRM_3(nn.Module):
    def __init__(self, mid_c = 64):
        super(FRM_3, self).__init__()
        self.tom_1 = MRB_2(mid_c)
        self.tod_1 = MRB_2(mid_c)
        self.tom_2 = MRB_1(mid_c)
        self.tod_2 = MRB_1(mid_c)
        self.relu = nn.ReLU(True)
    def forward(self, in_h, in_m, in_l, in_dh, in_d, in_dl):
        m_1 = self.tom_1(in_m, in_l, in_h)
        d_1 = self.tod_1(in_d, in_dl, in_dh)
        m_2 = self.tom_2(m_1, d_1)
        d_2 = self.tod_2(d_1, m_1)
        return m_2, d_2

    
class CB(nn.Module):
    def __init__(self, iC_list, oC_list):
        super(CB, self).__init__()
        ic0, ic1, ic2, ic3, ic4 = iC_list
        oc0, oc1, oc2, oc3, oc4 = oC_list
        self.conv1 = BasicConv2d(ic0, oc0, 3, 1, 1)
        self.conv2 = BasicConv2d(ic1, oc1, 3, 1, 1)
        self.conv3 = BasicConv2d(ic2, oc2, 3, 1, 1)
        self.conv4 = BasicConv2d(ic3, oc3, 3, 1, 1)
        self.conv5 = BasicConv2d(ic4, oc4, 3, 1, 1)

    def forward(self, *xs):
        out_xs = []
        out_xs.append(self.conv1(xs[0]))
        out_xs.append(self.conv2(xs[1]))
        out_xs.append(self.conv3(xs[2]))
        out_xs.append(self.conv4(xs[3]))
        out_xs.append(self.conv5(xs[4]))
        return out_xs

#FRM
class FRM(nn.Module):
    def __init__(self, ch):
        super(FRM, self).__init__()
        self.conv1 = FRM_h(ch)
        self.conv2 = FRM_3(ch)
        self.conv3 = FRM_3(ch)
        self.conv4 = FRM_3(ch)
        self.conv5 = FRM_l(ch)
    def forward(self, *xs):
        # in_data_2, in_data_4, in_data_8, in_data_16, in_data_32
        # in_depth_2, in_depth_4, in_depth_8, in_depth_16, in_depth_32
        out_xs = []
        out_xs.extend(self.conv1(       xs[0], xs[1],        xs[5], xs[6]))
        out_xs.extend(self.conv2(xs[0], xs[1], xs[2], xs[5], xs[6], xs[7]))
        out_xs.extend(self.conv3(xs[1], xs[2], xs[3], xs[6], xs[7], xs[8]))
        out_xs.extend(self.conv4(xs[2], xs[3], xs[4], xs[7], xs[8], xs[9]))
        out_xs.extend(self.conv5(xs[3], xs[4],        xs[8], xs[9]       ))
        #print(len(out_xs))
        return out_xs


class SFM(nn.Module):
    def __init__(self):
        super(SFM, self).__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv_1 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv_2 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv_3 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(64, 1, 1)

    def forward(self, in_data_m, in_data_l, attention):
        in_data_m = in_data_m + torch.mul(in_data_m, self.upsample2(attention.sigmoid()))
        in_data_l = self.upconv_1(self.upsample2(in_data_l))
        out_data_f = in_data_m * in_data_l
        out_data_f_m = self.upconv_2(out_data_f + in_data_m + in_data_l)
        out_data_f_l = self.upconv_3(out_data_f + in_data_l + in_data_m)
        out_map = self.classifier(out_data_f_l)
        return out_data_f_m, out_map