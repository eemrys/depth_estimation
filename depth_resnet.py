import torch.nn as nn
from functools import partial

from resnet.encoder import ResnetEncoder
from resnet.decoder import DepthDecoder
from resnet.layers import disp_to_depth


class DepthResNet(nn.Module):
    def __init__(self, version, **kwargs):
        super().__init__()

        num_layers = int(version[:2])
        pretrained = version[2:] == 'pt'
        assert num_layers in [18, 34, 50]

        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained)
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=100.0)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        disps = [x[('disp', i)] for i in range(4)]

        if self.training:
            return [self.scale_inv_depth(d)[0] for d in disps]
        else:
            return self.scale_inv_depth(disps[0])[0]
