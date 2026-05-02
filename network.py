

import torch
import torch.nn as nn
import torch.optim as optim
from mambapy.mamba import Mamba, MambaConfig
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ------------------------------
# 1. AMS-Fuse Block (PyTorch)
# ------------------------------
class AMSFuse(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(AMSFuse, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction
        self.num_branches = 3

        # Branch 1: 3x3 Conv
        self.branch1 = self._conv_block(in_channels, out_channels, 3, padding=1)

        # Branch 2: Asymmetric Conv (1x5 + 5x1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        # Branch 3: Dilated Convs (dilation=2 → 3)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(self.num_branches))

        # Channel recalibration gate
        total_channels = out_channels * self.num_branches
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_channels, total_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels // reduction, total_channels, kernel_size=1),
            nn.Sigmoid()
        )

        
        self.fusion_conv = self._conv_block(total_channels, out_channels, 1, padding=0)

        
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual = nn.Identity()

        self._initialize_weights()

    def _conv_block(self, in_ch, out_ch, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def _initialize_weights(self):
      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              # Fixed: Use 'relu' instead of 'gelu'
              nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
              if m.bias is not None:
                  nn.init.zeros_(m.bias)
          elif isinstance(m, nn.BatchNorm2d):
              nn.init.constant_(m.weight, 1)
              nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = self.residual(x)

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        branches = [b1, b2, b3]

        # Softmax-normalized scale weights
        scale_weights = torch.softmax(self.scale_weights, dim=0)
        scaled_branches = [scale_weights[i] * branch for i, branch in enumerate(branches)]

        fused = torch.cat(scaled_branches, dim=1)  # [B, 3*C, H, W]

        # Dynamic channel recalibration
        attention = self.channel_gate(fused)
        recalibrated = fused * attention

        # Final projection
        out = self.fusion_conv(recalibrated)

        # Residual connection
        out = out + residual
        return out


# ------------------------------
# 2. Patch Embedding (Unchanged)
# ------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_channels, patch_size=4, embed_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H//p, W//p)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x


# ------------------------------
# 3. Local Mamba (Restricted Context)
# ------------------------------
class LocalMamba(nn.Module):
    def __init__(self, d_model=256, local_window=8):
        super().__init__()
        self.local_window = local_window
        self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=1))

    def forward(self, x):
        B, N, C = x.shape
        out = torch.zeros_like(x)
        for start in range(0, N, self.local_window):
            end = min(start + self.local_window, N)
            out[:, start:end, :] = self.mamba(x[:, start:end, :])
        return out


# ------------------------------
# 4. CNN + Local-Global Mamba (Updated with AMSFuse)
# ------------------------------
class CNN_LocalGlobalMamba(nn.Module):
    def __init__(self, num_classes=8, patch_size=4, num_local_global_pairs=1):
        super().__init__()

        
        self.cnn = nn.Sequential(
            # Stage 1: 3 -> 32
            nn.Conv2d(3, 32, 3, 1, 1),
            #nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #AMSFuse(32, 32),
            nn.MaxPool2d(2),
            AMSFuse(32, 64),
            AMSFuse(64, 64),
            nn.MaxPool2d(2),
            AMSFuse(64, 96),
            AMSFuse(96, 128),
            nn.MaxPool2d(2),
            AMSFuse(128, 256),
            AMSFuse(256, 512),
            nn.ReLU(inplace=True), 
        )

        
        self.patch_embed = PatchEmbed(512, patch_size=patch_size, embed_dim=512)

        
        layers = []
        for _ in range(num_local_global_pairs):
            layers.append(LocalMamba(d_model=512))
            layers.append(Mamba(MambaConfig(d_model=512, n_layers=1)))
        self.mamba_layers = nn.ModuleList(layers)

       
        self.classifier = nn.Sequential(
              nn.Linear(512, 256),
              nn.ReLU(),
              nn.Linear(256, 64),
              nn.ReLU(),
              nn.Linear(64, num_classes)
            )

    def forward(self, x):
        x = self.cnn(x)  
        x = self.patch_embed(x)  

        for layer in self.mamba_layers:
            x = layer(x)

        x = x.mean(dim=1)  # Global average pooling 
        return self.classifier(x)