
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules.block as block

# Architecture Definitions
class HyperConv(nn.Module):
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, padding=k//2)
        self.hyper_gate = nn.Sequential(nn.Conv2d(c2, c2, 1), nn.Sigmoid())
    def forward(self, x): 
        feat = self.conv(x)
        return feat * self.hyper_gate(feat)

class HyperAttentionMasking(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels // 8, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(channels // 8, channels, 1, bias=False), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
    def forward(self, x):
        x_ca = x * self.ca(x)
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        mask = self.sa(torch.cat([avg_out, max_out], dim=1))
        return x_ca * mask

class ConsolidatedHyperBlock(block.Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        self.h_conv = HyperConv(c2, c2)
        self.ham = HyperAttentionMasking(c2)
    def forward(self, x):
        return self.ham(self.h_conv(super().forward(x)))

def load_hyper_yolo(weights_path):
    block.Bottleneck = ConsolidatedHyperBlock
    return YOLO(weights_path)
