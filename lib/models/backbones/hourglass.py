import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckX2(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(BottleneckX2, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes

        mid_planes = (out_planes // 2) if out_planes >= in_planes else in_planes // 2

        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_planes)

        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(mid_planes)

        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        if in_planes != out_planes:
            self.conv4 = nn.Conv2d(in_planes, out_planes, bias=True, kernel_size=1)

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

        if self.in_planes != self.out_planes:
            residual = self.conv4(x)

        out += residual
        out = self.relu(out)
        return out


class HourglassBisected(nn.Module):

    def __init__(self, block, nblocks, in_planes, depth=4):
        super(HourglassBisected, self).__init__()
        self.depth = depth
        self.hg = self._make_hourglass(block, nblocks, in_planes, depth)

    def _make_hourglass(self, block, nblocks, in_planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                _res = []
                if j == 1:
                    _res.append(self._make_residual(block, nblocks, in_planes))
                else:
                    _res.append(self._make_residual(block, nblocks, in_planes))
                    _res.append(self._make_residual(block, nblocks, in_planes))

                res.append(nn.ModuleList(_res))

            if i == 0:
                _res = []
                _res.append(self._make_residual(block, nblocks, in_planes))
                _res.append(self._make_residual(block, nblocks, in_planes))
                res.append(nn.ModuleList(_res))

            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _make_residual(self, block, nblocks, in_planes):
        layers = []
        for i in range(0, nblocks):
            layers.append(block(in_planes, in_planes))
        return nn.Sequential(*layers)

    def _hourglass_foward(self, n, x):
        up1_1 = self.hg[n - 1][0][0](x)
        up1_2 = self.hg[n - 1][0][1](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1][0](low1)

        if n > 1:
            low2_1, low2_2, latent = self._hourglass_foward(n - 1, low1)
        else:
            latent = low1
            low2_1 = self.hg[n - 1][3][0](low1)
            low2_2 = self.hg[n - 1][3][1](low1)

        low3_1 = self.hg[n - 1][2][0](low2_1)
        low3_2 = self.hg[n - 1][2][1](low2_2)

        up2_1 = F.interpolate(low3_1, scale_factor=2)
        up2_2 = F.interpolate(low3_2, scale_factor=2)
        out_1 = up1_1 + up2_1
        out_2 = up1_2 + up2_2

        return out_1, out_2, latent

    def forward(self, x):
        return self._hourglass_foward(self.depth, x)
