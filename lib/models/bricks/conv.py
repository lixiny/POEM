import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channel,
        out_chanel,
        kernel_size=3,
        padding=1,
        stride=1,
        dilation=1,
        relu=False,
        norm='bn',
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel,
                              out_chanel,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride,
                              dilation=dilation)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_chanel)
        else:
            self.norm = None
        self.relu = nn.ReLU(True) if relu else None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu is not None:
            x = self.relu(x)

        return x