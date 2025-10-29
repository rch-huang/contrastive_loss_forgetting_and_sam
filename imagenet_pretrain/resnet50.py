import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- helpers to mirror printed structure ----

class BatchNormRelu(nn.Sequential):
    def __init__(self, num_features, relu=True):
        layers = [nn.BatchNorm2d(num_features)]
        layers.append(nn.ReLU(inplace=True) if relu else nn.Identity())
        super().__init__(*layers)

class Stem(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            BatchNormRelu(64, relu=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

class Projection(nn.Module):
    """1×1 projection shortcut with BN (no ReLU)."""
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        self.bn = BatchNormRelu(out_ch, relu=False)

    def forward(self, x):
        return self.bn(self.shortcut(x))

class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, in_ch, mid_ch, stride=1, use_projection=False):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, bias=False),
            BatchNormRelu(mid_ch, relu=True),

            # v1.5 style: stride on the 3×3
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            BatchNormRelu(mid_ch, relu=True),

            nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, bias=False),
            BatchNormRelu(out_ch, relu=False),
        )
        self.projection = Projection(in_ch, out_ch, stride) if use_projection else nn.Identity()

    def forward(self, x):
        y = self.net(x)
        shortcut = self.projection(x) if not isinstance(self.projection, nn.Identity) else x
        return F.relu(y + shortcut, inplace=True)

class Blocks(nn.Module):
    def __init__(self, in_ch, mid_ch, num_blocks, first_stride):
        super().__init__()
        blocks = [BottleneckBlock(in_ch, mid_ch, stride=first_stride, use_projection=True)]
        out_ch = mid_ch * BottleneckBlock.expansion
        for _ in range(num_blocks - 1):
            blocks.append(BottleneckBlock(out_ch, mid_ch, stride=1, use_projection=False))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

# ---- ResNet-50 backbone without fc ----

class ResNet50Backbone(nn.Module):
    """
    Same structure as your printed ResNet-50, except no final fc.
    Returns a [B, 2048] embedding (after GAP + flatten).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            Stem(),                                # -> 64, H/4
            Blocks(in_ch=64,   mid_ch=64,  num_blocks=3, first_stride=1),  # -> 256, H/4
            Blocks(in_ch=256,  mid_ch=128, num_blocks=4, first_stride=2),  # -> 512, H/8
            Blocks(in_ch=512,  mid_ch=256, num_blocks=6, first_stride=2),  # -> 1024, H/16
            Blocks(in_ch=1024, mid_ch=512, num_blocks=3, first_stride=2),  # -> 2048, H/32
        )
        self.feature_dim = 2048  # convenient attribute

    def forward(self, x):
        x = self.net(x)                          # [B, 2048, H/32, W/32]
        x = F.adaptive_avg_pool2d(x, (1, 1))     # [B, 2048, 1, 1]
        x = torch.flatten(x, 1)                  # [B, 2048]
        return x