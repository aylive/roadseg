"""Deep Layer Aggregation
- DLA structure Adapted from Song's work of Sat2Graph.
- Differs from the version of `timm`.
"""
from typing import Dict, Callable

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Resnet(nn.Module):

    def __init__(self, ch, resnet_step=0) -> None:
        super(Resnet, self).__init__()

        self.res_blocks = [
            self._res_block(ch, downsample=False) for _ in range(resnet_step)
        ] if resnet_step > 0 else []
        self.batch_norm = nn.BatchNorm2d(ch, momentum=0.01)

    def _res_block(self, ch, use_bias=True, downsample=True) -> Dict[str, Callable]:
        modules = [
            nn.BatchNorm2d(ch, momentum=0.01),
            nn.ReLU(inplace=True),
        ]
        modules_init = []

        if downsample:
            modules.append(nn.Conv2d(ch, ch, kernel_size=3, stride=2, bias=use_bias, padding=3//2))
            modules_init.append(nn.Conv2d(ch, ch, kernel_size=1, stride=2, bias=use_bias, padding=0))
        else:
            modules.append(nn.Conv2d(ch, ch, kernel_size=3, stride=1, bias=use_bias, padding=3//2))

        modules.append(nn.BatchNorm2d(ch, momentum=0.01))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Conv2d(ch, ch, kernel_size=3, stride=1, bias=use_bias, padding=3//2))
        
        return {
            'res': nn.Sequential(*modules),
            'init': nn.Sequential(*modules_init),
        }

    def forward(self, x: torch.Tensor):
        for blk in self.res_blocks:
            x_init = blk['init'].to(x.device)(x)
            x_res = blk['res'].to(x.device)(x)

            x = x_init + x_res
        
        x = self.batch_norm(x)
        return F.relu(x)


class DLAWithResnet(nn.Module):

    def __init__(
            self, 
            in_chans,
            out_chans,
            base_chans = 12,
            resnet_step = 8,
        ) -> None:
        super(DLAWithResnet, self).__init__()

        self.base_layer1 = \
            self._conv_layer(in_chans, base_chans, ks=5, stride=1, batchnorm=False)
        self.base_layer2 = \
            self._conv_layer(base_chans, base_chans*2, ks=5, stride=2)
        
        ### level0
        reduce_block = self._reduce_block(base_chans*2, base_chans*4)
        resnet_block = self._resnet_block(base_chans*4, resnet_step=int(resnet_step/8))
        self.x_4s = nn.Sequential(*list(reduce_block), resnet_block)

        reduce_block = self._reduce_block(base_chans*4, base_chans*8)
        resnet_block = self._resnet_block(base_chans*8, resnet_step=int(resnet_step/4))
        self.x_8s = nn.Sequential(*list(reduce_block), resnet_block)

        reduce_block = self._reduce_block(base_chans*8, base_chans*16)
        resnet_block = self._resnet_block(base_chans*16, resnet_step=int(resnet_step/2))
        self.x_16s = nn.Sequential(*list(reduce_block), resnet_block)

        reduce_block = self._reduce_block(base_chans*16, base_chans*32)
        resnet_block = self._resnet_block(base_chans*32, resnet_step=resnet_step)
        self.x_32s = nn.Sequential(*list(reduce_block), resnet_block)

        ### level1        
        self.a1_2s = self._aggregate_block(base_chans*2, base_chans*4, base_chans*4)
        self.a1_4s = self._aggregate_block(base_chans*4, base_chans*8, base_chans*8)
        self.a1_8s = self._aggregate_block(base_chans*8, base_chans*16, base_chans*16)
        self.a1_16s = self._aggregate_block(base_chans*16, base_chans*32, base_chans*32)
        
        resnet_block = self._resnet_block(base_chans*32, resnet_step=int(resnet_step/2))
        self.a1_16s['resnet_block'] = resnet_block

        ### level2
        self.a2_2s = self._aggregate_block(base_chans*4, base_chans*8, base_chans*4)
        self.a2_4s = self._aggregate_block(base_chans*8, base_chans*16, base_chans*8)
        self.a2_8s = self._aggregate_block(base_chans*16, base_chans*32, base_chans*16)

        resnet_block = self._resnet_block(base_chans*16, resnet_step=int(resnet_step/4))
        self.a2_8s['resnet_block'] = resnet_block

        ### level3
        self.a3_2s = self._aggregate_block(base_chans*4, base_chans*8, base_chans*4)
        self.a3_4s = self._aggregate_block(base_chans*8, base_chans*16, base_chans*8)
        
        resnet_block = self._resnet_block(base_chans*8, resnet_step=int(resnet_step/8))
        self.a3_4s['resnet_block'] = resnet_block

        ### level4
        self.a4_2s = self._aggregate_block(base_chans*4, base_chans*8, base_chans*8)
        
        ### level5
        self.a5_2s = self._conv_layer(base_chans*8, base_chans*4, ks=3, stride=1)

        ### out
        self.a_out = \
            self._aggregate_block(base_chans, base_chans*4, base_chans*4, batchnorm=False)
        
        self.a_out['final'] = \
            self._conv_layer(base_chans*4, out_chans, ks=3, stride=1, batchnorm=False, activation='linear')

    def _conv_layer(
            self, in_ch, out_ch, ks, stride, padding='same', batchnorm=True, activation='relu',
        ):
        modules = [
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=ks//2)
        ]
        
        if batchnorm:
            modules.append(nn.BatchNorm2d(out_ch, momentum=0.01))
        
        if activation == 'relu':
            modules.append(nn.ReLU(inplace=True))
        elif activation == 'linear':
            modules.append(nn.Identity())

        return nn.Sequential(*modules)
    
    def _deconv_layer(
            self, in_ch, out_ch, ks, stride, padding=1, output_padding=1, batchnorm=True, activation='relu',
        ):
        modules = [nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=ks, stride=stride, padding=padding, output_padding=output_padding,
        )]
        
        if batchnorm:
            modules.append(nn.BatchNorm2d(out_ch, momentum=0.01))
        
        if activation == 'relu':
            modules.append(nn.ReLU(inplace=True))
        elif activation == 'linear':
            modules.append(nn.Identity())

        return nn.Sequential(*modules)
    
    def _reduce_block(self, in_ch, out_ch):
        conv1 = self._conv_layer(in_ch, in_ch, ks=3, stride=1)
        conv2 = self._conv_layer(in_ch, out_ch, ks=3, stride=2)
        return nn.Sequential(*(list(conv1) + list(conv2)))

    def _resnet_block(self, ch, resnet_step=0):
        return Resnet(ch, resnet_step=resnet_step)

    def _aggregate_block(self, in_ch1, in_ch2, out_ch, batchnorm=True) -> Dict[str, Callable]:
        modules_x2 = self._deconv_layer(in_ch2, in_ch2, ks=3, stride=2, batchnorm=batchnorm)
        
        conv1 = self._conv_layer(in_ch1+in_ch2, in_ch1+in_ch2, ks=3, stride=1, batchnorm=batchnorm)
        conv2 = self._conv_layer(in_ch1+in_ch2, out_ch, ks=3, stride=1, batchnorm=batchnorm)
        modules_x = nn.Sequential(*(list(conv1) + list(conv2)))

        return {'x': modules_x, 'x2': modules_x2}
    
    def forward(self, input: torch.Tensor):
        conv1 = self.base_layer1(input)
        conv2 = self.base_layer2(conv1)

        x_4s = self.x_4s(conv2)
        x_8s = self.x_8s(x_4s)
        x_16s = self.x_16s(x_8s)
        x_32s = self.x_32s(x_16s)

        def agg(x1, x2, agg_func: Dict[str, Callable]):
            x2 = agg_func['x2'].to(x2.device)(x2)
            x = torch.cat([x1, x2], dim=1)
            x = agg_func['x'].to(x.device)(x)
            return x
        
        a1_2s = agg(conv2, x_4s, self.a1_2s)
        a1_4s = agg(x_4s, x_8s, self.a1_4s)
        a1_8s = agg(x_8s, x_16s, self.a1_8s)
        a1_16s = agg(x_16s, x_32s, self.a1_16s)
        a1_16s = self.a1_16s['resnet_block'].to(a1_16s.device)(a1_16s)

        a2_2s = agg(a1_2s, a1_4s, self.a2_2s)
        a2_4s = agg(a1_4s, a1_8s, self.a2_4s)
        a2_8s = agg(a1_8s, a1_16s, self.a2_8s)
        a2_8s = self.a2_8s['resnet_block'].to(a2_8s.device)(a2_8s)

        a3_2s = agg(a2_2s, a2_4s, self.a3_2s)
        a3_4s = agg(a2_4s, a2_8s, self.a3_4s)
        a3_4s = self.a3_4s['resnet_block'].to(a3_4s.device)(a3_4s)

        a4_2s = agg(a3_2s, a3_4s, self.a4_2s)

        a5_2s = self.a5_2s.to(a4_2s.device)(a4_2s)

        a_out = agg(conv1, a5_2s, self.a_out)
        a_out = self.a_out['final'].to(a_out.device)(a_out)
        a_out = F.softmax(a_out, dim=1) # required by smp scheme
        return a_out