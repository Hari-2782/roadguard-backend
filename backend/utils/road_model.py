"""
SimpleFastSCNN - Lightweight Road Segmentation Architecture
Matches the architecture used to train models/road_segmentation/best.pth
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, pad=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class _InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand=6):
        super().__init__()
        mid = in_ch * expand
        self.use_res = stride == 1 and in_ch == out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        if self.use_res:
            return x + self.block(x)
        return self.block(x)


class SimpleFastSCNN(nn.Module):
    """
    Fast Semantic Segmentation Network (Simplified variant).
    Input: (B, 3, H, W)
    Output: (B, 1, H, W) — sigmoid road mask
    """
    def __init__(self, num_classes=1):
        super().__init__()

        # Learning to Downsample (stride 8 total)
        self.ld = nn.Sequential(
            _ConvBNReLU(3, 32, 3, stride=2),
            _DSConv(32, 48, stride=2),
            _DSConv(48, 64, stride=2)
        )

        # Global Feature Extractor
        self.gfe = nn.Sequential(
            _InvertedResidual(64, 64),
            _InvertedResidual(64, 64),
            _InvertedResidual(64, 96, stride=2),
            _InvertedResidual(96, 96),
            _InvertedResidual(96, 128, stride=2),
            _InvertedResidual(128, 128),
        )
        # PPM-style global context
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool_conv = _ConvBNReLU(128, 128, 1, pad=0)

        # Feature Fusion Module
        self.ffm_ld = _ConvBNReLU(64, 128, 1, pad=0)
        self.ffm_gfe = _ConvBNReLU(128, 128, 1, pad=0)
        self.ffm_out = nn.Sequential(
            _DSConv(128, 128),
            _DSConv(128, 128)
        )

        # Classifier head
        self.cls = nn.Sequential(
            _DSConv(128, 128),
            _DSConv(128, 128),
            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        ld = self.ld(x)                                         # /8 size
        gfe = self.gfe(ld)                                      # /32 size
        # Global context injection
        ctx = self.pool(gfe)
        ctx = self.pool_conv(ctx)
        gfe = gfe + F.interpolate(ctx, size=gfe.shape[2:], mode='bilinear', align_corners=False)
        # Feature fusion
        gfe_up = F.interpolate(gfe, size=ld.shape[2:], mode='bilinear', align_corners=False)
        fused = self.ffm_ld(ld) + self.ffm_gfe(gfe_up)
        fused = self.ffm_out(fused)
        # Classify and upsample to original resolution
        out = self.cls(fused)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return torch.sigmoid(out)
