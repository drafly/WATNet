

## Learning Enriched Features for Fast Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
## https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/

# --- Imports --- #
import torch
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as stx
import math
from models.max_models.maxvit import MaxViTTransformerBlock,grid_partition,grid_reverse
from models.WT.transform import DWT,IWT
from pytorch_wavelets import DWTForward, DWTInverse
import functools


# Spatial Attention Layer
class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # torch.max will output 2 things, and we want the 1st one
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)  # [N,2,H,W]  could add 1x1 conv -> [N,3,H,W]
        y = self.conv_du(channel_pool)

        return x * y

##########################################################################
# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)




#################################################################################
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


########################################################################

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)



########################################################################
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=8):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = LayerNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
            act,
            nn.Conv2d(inp, inp, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        )

        self.act = act

    def forward(self, x):
        identity = x
        x = self.body(x)

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class HWB(nn.Module):
    def __init__(self, n_feat, o_feat, kernel_size, reduction, bias, act):
        super(HWB, self).__init__()
        self.h_feat = int(n_feat * 0.5)

        modules_body = \
            [
                conv(self.h_feat, n_feat, kernel_size, bias=bias),
                act,
                conv(n_feat, self.h_feat, kernel_size, bias=bias)
            ]
        self.body = nn.Sequential(*modules_body)

        self.Coo = CoordAtt(self.h_feat, self.h_feat, 4)

        self.conv1x1 = nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=1, bias=bias)
        self.conv3x3 = nn.Conv2d(n_feat, o_feat, kernel_size=3, padding=1, bias=bias)
        self.activate = act
        self.conv1x1_final = nn.Conv2d(n_feat, o_feat, kernel_size=1, bias=bias)

        self.dwt = DWTForward(J=1, mode='zero', wave='db2').cuda()  # Accepts all wave types available to PyWavelets
        self.iwt = DWTInverse(mode='zero', wave='db2').cuda()

    def forward(self, x):
        residual = x

        # Split 2 part
        wavelet_path_in, identity_path = torch.chunk(x, 2, dim=1)

        # Wavelet domain (Dual attention)
        ## return a tuple
        x_dwt = self.dwt(wavelet_path_in)

        x_1 = x_dwt[0]
        x_2 = x_dwt[1][0]

        B, C, H, W = x_2.shape[0], x_2.shape[1] * x_2.shape[2], x_2.shape[3], x_2.shape[4]
        x_2_new = x_2.view(B, C, H, W)
        # y_dwt = torch.cat((x_1, x_2_new), dim=1)

        #res = self.body(x_2_new)

        res = self.Coo(x_1) + x_1

        # res = self.conv1x1(res) + y_dwt

        # y1, y2 = torch.split(res, split_size_or_sections=[8, 24], dim=1)
        #y2 = [res.view(B,C//3,3,H,W)]

        wavelet_path = self.iwt((res, [x_2]))

        out = torch.cat([wavelet_path, identity_path], dim=1)
        out = self.activate(self.conv3x3(out))
        out += self.conv1x1_final(residual)

        return out






##########################################################################
# Half Wavelet Dual Attention Block (HWB)
# class HWB(nn.Module):
#     def __init__(self, n_feat, o_feat, kernel_size, reduction, bias, act):
#         super(HWB, self).__init__()
#         # self.dwt = DWT()
#         # self.iwt = IWT()
#
#         modules_body = \
#             [
#                 conv(n_feat*2, n_feat, kernel_size, bias=bias),
#                 act,
#                 conv(n_feat, n_feat*2, kernel_size, bias=bias)
#             ]
#         self.body = nn.Sequential(*modules_body)
#
#         self.WSA = SALayer()
#         self.WCA = CALayer(n_feat*2, reduction, bias=bias)
#
#         self.conv1x1 = nn.Conv2d(n_feat*4, n_feat*2, kernel_size=1, bias=bias)
#         self.conv3x3 = nn.Conv2d(n_feat, o_feat, kernel_size=3, padding=1, bias=bias)
#         self.activate = act
#         self.conv1x1_final = nn.Conv2d(n_feat, o_feat, kernel_size=1, bias=bias)
#
#         self.dwt = DWTForward(J=1, mode='zero', wave='db2').cuda() # Accepts all wave types available to PyWavelets
#         self.iwt = DWTInverse(mode='zero', wave='db2').cuda()
#
#
#     def forward(self, x):
#         residual = x
#
#         # Split 2 part
#         wavelet_path_in, identity_path = torch.chunk(x, 2, dim=1)
#
#         # Wavelet domain (Dual attention)
#         ## return a tuple
#         x_dwt = self.dwt(wavelet_path_in)
#
#         x_1 =x_dwt[0]
#         x_2 =x_dwt[1][0]
#
#         B,C,H,W = x_2.shape[0],x_2.shape[1]*x_2.shape[2],x_2.shape[3],x_2.shape[4]
#         x_2_new = x_2.view(B,C,H,W)
#
#         y_dwt = torch.cat((x_1, x_2_new), dim=1)
#
#         res = self.body(y_dwt)
#         branch_sa = self.WSA(res)
#         branch_ca = self.WCA(res)
#         res = torch.cat([branch_sa, branch_ca], dim=1)
#         res = self.conv1x1(res) + y_dwt
#
#         y1, y2 = torch.split(res, split_size_or_sections=[8, 24], dim=1)
#         y2 = [y2.view(B,C//3,3,H,W)]
#
#         wavelet_path = self.iwt((y1,y2))
#
#
#         out = torch.cat([wavelet_path, identity_path], dim=1)
#         out = self.activate(self.conv3x3(out))
#         out += self.conv1x1_final(residual)
#
#         return out
#
#
#



class Interpolate(nn.Module):

    def __init__(self, channel: int, scale_factor: int):
        super().__init__()
        # assert 'mode' not in kwargs and 'align_corners' not in kwargs and 'size' not in kwargs
        assert isinstance(scale_factor, int) and scale_factor > 1 and scale_factor % 2 == 0
        self.scale_factor = scale_factor
        kernel_size = scale_factor + 1  # keep kernel size being odd
        self.weight = nn.Parameter(
            torch.empty((1, 1, kernel_size, kernel_size), dtype=torch.float32).expand(channel, -1, -1, -1)
        )
        self.conv = functools.partial(
            F.conv2d, weight=self.weight, bias=None, padding=scale_factor // 2, groups=channel
        )
        with torch.no_grad():
            self.weight.fill_(1 / (kernel_size * kernel_size))

    def forward(self, t):
        if t is None:
            return t
        return self.conv(F.interpolate(t, scale_factor=self.scale_factor, mode='nearest'))

    @staticmethod
    def naive(t: torch.Tensor, size: Tuple[int, int], **kwargs):
        if t is None or t.shape[2:] == size:
            return t
        else:
            assert 'mode' not in kwargs and 'align_corners' not in kwargs
            return F.interpolate(t, size, mode='nearest', **kwargs)


##########################################################################
##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3,reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats    = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        # L1 +  L2的结果

        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)

        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)
        #fcs应该不止一个，里面有很多个kernel大小为1的conv2d
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()

        attention_vectors = self.softmax(attention_vectors)

        #将s1,s2分别乘以L1和L2
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V        



class ContextBlock(nn.Module):

    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.modeling(x)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term

        return x

##########################################################################
### --------- Residual Context Block (RCB) ----------
class RCB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCB, self).__init__()
        
        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act, 
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups)
        )

        self.act = act
        
        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.act(self.gcnet(res))
        res += x
        return res


##########################################################################
##---------- Resizing Modules ----------    
class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels*chan_factor), 1, stride=1, padding=0, bias=bias)
            )

    def forward(self, x):
        return self.bot(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x) #[b,n_feats,H,W]
        return x         #[b,n_feats* ratio ,H/2,W/2]

class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=True):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels//chan_factor), 1, stride=1, padding=0, bias=bias),

            Interpolate(channel=int(in_channels//chan_factor), scale_factor=2)
            )

    def forward(self, x):
        return self.bot(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


##########################################################################
##---------- Multi-Scale Resiudal Block (MRB) ----------
class MRB(nn.Module):
    def __init__(self, n_feat, height, width, chan_factor, bias,groups):
        super(MRB, self).__init__()

        self.n_feat, self.height, self.width = n_feat, height, width

        self.dau_top = RCB(int(n_feat*chan_factor**0), bias=bias, groups=groups)
        #self.dau_top = HWB(int(n_feat * chan_factor ** 0),int(n_feat * chan_factor ** 0),reduction=8,kernel_size=3, bias=bias,act=nn.PReLU() )
        # self.max_top = MaxViTTransformerBlock(in_channels=int(n_feat*chan_factor**0), partition_function=grid_partition,
        #                                         reverse_function=grid_reverse,
        #                                         num_heads=2,  # 32
        #                                         grid_window_size=(8, 8),  # (7,7)
        #                                         attn_drop=0.,
        #                                         drop=0.,
        #                                         drop_path=0.,
        #                                         mlp_ratio=4.,
        #                                         act_layer=nn.GELU,
        #                                         norm_layer=nn.LayerNorm)
        self.dau_mid = RCB(int(n_feat*chan_factor**1), bias=bias, groups=groups)
        #self.dau_mid = HWB(int(n_feat * chan_factor ** 1),int(n_feat * chan_factor ** 1),reduction=8,kernel_size=3, bias=bias,act=nn.PReLU() )
        self.max_mid= MaxViTTransformerBlock(in_channels=int(n_feat * chan_factor ** 1),
                                              partition_function=grid_partition,
                                              reverse_function=grid_reverse,
                                              num_heads=4,  # 32
                                              grid_window_size=(4, 4),  # (7,7)
                                              attn_drop=0.,
                                              drop=0.,
                                              drop_path=0.,
                                              mlp_ratio=4.,
                                              act_layer=nn.GELU,
                                              norm_layer=nn.LayerNorm)
        self.dau_bot = RCB(int(n_feat*chan_factor**2), bias=bias, groups=groups)
        #self.dau_bot = HWB(int(n_feat * chan_factor ** 2),int(n_feat * chan_factor ** 2),reduction=8,kernel_size=3, bias=bias,act=nn.PReLU() )
        self.max_bot = MaxViTTransformerBlock(in_channels=int(n_feat * chan_factor ** 2),
                                              partition_function=grid_partition,
                                              reverse_function=grid_reverse,
                                              num_heads=2,  # 32
                                              grid_window_size=(2, 2),  # (7,7)
                                              attn_drop=0.,
                                              drop=0.,
                                              drop_path=0.,
                                              mlp_ratio=4.,
                                              act_layer=nn.GELU,
                                              norm_layer=nn.LayerNorm)

        self.down2 = DownSample(int((chan_factor**0)*n_feat),2,chan_factor)
        self.down4 = nn.Sequential(
            DownSample(int((chan_factor**0)*n_feat),2,chan_factor), 
            DownSample(int((chan_factor**1)*n_feat),2,chan_factor)
        )

        self.up21_1 = UpSample(int((chan_factor**1)*n_feat),2,chan_factor)
        self.up21_2 = UpSample(int((chan_factor**1)*n_feat),2,chan_factor)
        self.up32_1 = UpSample(int((chan_factor**2)*n_feat),2,chan_factor)
        self.up32_2 = UpSample(int((chan_factor**2)*n_feat),2,chan_factor)

        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, bias=bias)

        # only two inputs for SKFF
        self.skff_top = SKFF(int(n_feat*chan_factor**0), 2)
        self.skff_mid = SKFF(int(n_feat*chan_factor**1), 2)

    def forward(self, x):
        x_top = x.clone() #[B,n_feats,H,w]
        x_mid = self.down2(x_top)    #B,n_feats * ratio ,H/2,w/2]
        x_bot = self.down4(x_top)    #B,n_feats * ratio*ratio ,H/4,w/4]

        x_top = self.dau_top(x_top) #[B,n_feats,H,w]
        x_top_max = x_top

        x_mid = self.dau_mid(x_mid)
        x_mid_max = self.max_mid(x_mid) +x_mid

        x_bot = self.dau_bot(x_bot)#B,n_feats * ratio*ratio ,H/4,w/4]
        x_bot_max = self.max_bot(x_bot) + x_bot
        
        x_bot = x_bot_max
        x_mid = self.skff_mid([x_mid_max, self.up32_1(x_bot_max)])
        x_top = self.skff_top([x_top_max, self.up21_1(x_mid)])

        out = self.conv_out(x_top)
        out = out + x

        return out

##########################################################################
##---------- Recursive Residual Group (RRG) ----------
class RRG(nn.Module):
    def __init__(self, n_feat, n_MRB, height, width, chan_factor, bias=False, groups=1):
        super(RRG, self).__init__()
        modules_body = [MRB(n_feat, height, width, chan_factor, bias, groups) for _ in range(n_MRB)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
##---------- MIRNet  -----------------------
class MIRNet_v2(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        n_feat=80,
        chan_factor=1.5,
        n_RRG=4,
        n_MRB=2,
        height=3,
        width=2,
        scale=1,
        bias=False,
        task= None
    ):
        super(MIRNet_v2, self).__init__()

        kernel_size=3
        self.task = task

        self.conv_in = nn.Conv2d(inp_channels, n_feat, kernel_size=3, padding=1, bias=bias)
        
        self.hwb = HWB(n_feat,inp_channels,reduction=8,kernel_size=3, bias=bias,act=nn.PReLU() )

        modules_body = []
        
        modules_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=1))
#        modules_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=2))
#        modules_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=4))
#        modules_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=4))

        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=bias)
        

    def forward(self, inp_img):
        #[B,n_feats,H,W]
        shallow_feats = self.conv_in(inp_img) #[b,n,H,w]
        deep_feats = self.body(shallow_feats) #[b,n,H,w]

        if self.task == 'defocus_deblurring':
            deep_feats += shallow_feats
            out_img = self.conv_out(deep_feats)

        # else:
        #     out_img = self.conv_out(deep_feats)
        #     out_img += inp_img
        else:
            out_img = self.hwb(deep_feats)
            out_img += inp_img

        return out_img
