"""Deep Layer Aggregation
- DLA structure Adapted from Song's work of Sat2Graph.
- Differs from the version of `timm`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    def __init__(self, ch, use_bias=True, downsample=True) -> None:
        super().__init__()

        # Pack layers of resnet block into modules
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
        
        # Initialize the modules
        self.res_layer = nn.Sequential(*modules)
        self.init_layer = nn.Sequential(*modules_init)

    def forward(self, x: torch.Tensor):
        x_init = self.init_layer(x)
        x_res = self.res_layer(x)
        return x_init + x_res

class Resnet(nn.Module):

    def __init__(self, ch, resnet_step=0) -> None:
        super(Resnet, self).__init__()

        res_blocks = [
            ResBlock(ch, downsample=False) for _ in range(resnet_step)
        ] if resnet_step > 0 else []
        self.res_blocks = nn.Sequential(*res_blocks)
        self.batch_norm = nn.BatchNorm2d(ch, momentum=0.01)

    def forward(self, x: torch.Tensor):
        x = self.res_blocks(x)
        x = self.batch_norm(x)
        return F.relu(x)

class AggBlock(nn.Module):

    def __init__(self, in_ch1, in_ch2, out_ch, batchnorm=True) -> None:
        super().__init__()
        
        # Layer x2 passes
        self.x2 = deconv_layer(in_ch2, in_ch2, ks=3, stride=2, batchnorm=batchnorm)

        # Layer x1 passes
        conv1 = conv_layer(in_ch1+in_ch2, in_ch1+in_ch2, ks=3, stride=1, batchnorm=batchnorm)
        conv2 = conv_layer(in_ch1+in_ch2, out_ch, ks=3, stride=1, batchnorm=batchnorm)
        self.x1 = nn.Sequential(*(list(conv1) + list(conv2)))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x2 = self.x2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.x1(x)
        return x

class DLAWithResnet(nn.Module):

    def __init__(
            self, 
            in_chans,
            out_chans,
            base_chans = 12,
            resnet_step = 8,
        ) -> None:
        super().__init__()

        self.base_layer1 = \
            conv_layer(in_chans, base_chans, ks=5, stride=1, batchnorm=False)
        self.base_layer2 = \
            conv_layer(base_chans, base_chans*2, ks=5, stride=2)
        
        ### level0
        reduce_block = self._reduce_block(base_chans*2, base_chans*4)
        resnet_block = Resnet(base_chans*4, resnet_step=int(resnet_step/8))
        self.x_4s = nn.Sequential(*list(reduce_block), resnet_block)

        reduce_block = self._reduce_block(base_chans*4, base_chans*8)
        resnet_block = Resnet(base_chans*8, resnet_step=int(resnet_step/4))
        self.x_8s = nn.Sequential(*list(reduce_block), resnet_block)

        reduce_block = self._reduce_block(base_chans*8, base_chans*16)
        resnet_block = Resnet(base_chans*16, resnet_step=int(resnet_step/2))
        self.x_16s = nn.Sequential(*list(reduce_block), resnet_block)

        reduce_block = self._reduce_block(base_chans*16, base_chans*32)
        resnet_block = Resnet(base_chans*32, resnet_step=resnet_step)
        self.x_32s = nn.Sequential(*list(reduce_block), resnet_block)

        ### level1        
        self.a1_2s = AggBlock(base_chans*2, base_chans*4, base_chans*4)
        self.a1_4s = AggBlock(base_chans*4, base_chans*8, base_chans*8)
        self.a1_8s = AggBlock(base_chans*8, base_chans*16, base_chans*16)
        self.a1_16s = AggBlock(base_chans*16, base_chans*32, base_chans*32)
        
        self.a1_16s_resnet = Resnet(base_chans*32, resnet_step=int(resnet_step/2))

        ### level2
        self.a2_2s = AggBlock(base_chans*4, base_chans*8, base_chans*4)
        self.a2_4s = AggBlock(base_chans*8, base_chans*16, base_chans*8)
        self.a2_8s = AggBlock(base_chans*16, base_chans*32, base_chans*16)

        self.a2_8s_resnet = Resnet(base_chans*16, resnet_step=int(resnet_step/4))

        ### level3
        self.a3_2s = AggBlock(base_chans*4, base_chans*8, base_chans*4)
        self.a3_4s = AggBlock(base_chans*8, base_chans*16, base_chans*8)
        
        self.a3_4s_resnet = Resnet(base_chans*8, resnet_step=int(resnet_step/8))

        ### level4
        self.a4_2s = AggBlock(base_chans*4, base_chans*8, base_chans*8)
        
        ### level5
        self.a5_2s = conv_layer(base_chans*8, base_chans*4, ks=3, stride=1)

        ### out
        self.a_out = AggBlock(base_chans, base_chans*4, base_chans*4, batchnorm=False)
        
        self.output = conv_layer(
            base_chans*4, out_chans, ks=3, stride=1, batchnorm=False, activation='linear')
    
    def _reduce_block(self, in_ch, out_ch):
        conv1 = conv_layer(in_ch, in_ch, ks=3, stride=1)
        conv2 = conv_layer(in_ch, out_ch, ks=3, stride=2)
        return nn.Sequential(*(list(conv1) + list(conv2)))
    
    def forward(self, input: torch.Tensor):
        conv1 = self.base_layer1(input)
        conv2 = self.base_layer2(conv1)

        x_4s = self.x_4s(conv2)
        x_8s = self.x_8s(x_4s)
        x_16s = self.x_16s(x_8s)
        x_32s = self.x_32s(x_16s)
        
        a1_2s = self.a1_2s(conv2, x_4s)
        a1_4s = self.a1_4s(x_4s, x_8s)
        a1_8s = self.a1_8s(x_8s, x_16s)
        a1_16s = self.a1_16s(x_16s, x_32s)
        a1_16s = self.a1_16s_resnet(a1_16s)

        a2_2s = self.a2_2s(a1_2s, a1_4s)
        a2_4s = self.a2_4s(a1_4s, a1_8s)
        a2_8s = self.a2_8s(a1_8s, a1_16s)
        a2_8s = self.a2_8s_resnet(a2_8s)

        a3_2s = self.a3_2s(a2_2s, a2_4s)
        a3_4s = self.a3_4s(a2_4s, a2_8s)
        a3_4s = self.a3_4s_resnet(a3_4s)

        a4_2s = self.a4_2s(a3_2s, a3_4s)

        a5_2s = self.a5_2s(a4_2s)

        a_out = self.a_out(conv1, a5_2s)
        a_out = self.output(a_out)
        a_out = F.sigmoid(a_out)
        return a_out

def conv_layer(
        in_ch, out_ch, ks, stride, padding='same', batchnorm=True, activation='relu',
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

def deconv_layer(
        in_ch, out_ch, ks, stride, padding=1, output_padding=1, batchnorm=True, activation='relu',
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

# Test the file
if __name__ == "__main__":
    x = torch.rand(4, 3, 352, 352)
    model = DLAWithResnet(in_chans=3, out_chans=2)
    output = model(x)
    print(output[0].sum(dim=0))