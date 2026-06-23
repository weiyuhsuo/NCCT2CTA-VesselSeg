"""Multi-task U-Net with shared encoder and dual decoders.

Shared Encoder: extracts features from NCCT input
CTA Decoder: reconstructs CTA image (regression, tanh output)
SEG Decoder: predicts vessel segmentation mask (sigmoid output)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    """U-Net encoder with max-pooling downsampling."""

    def __init__(self, in_channels, features=(64, 128, 256, 512)):
        super().__init__()
        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for feat in features:
            self.encoders.append(ConvBlock(prev_ch, feat))
            prev_ch = feat
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """Returns list of skip features (before each pool) and bottleneck."""
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        return x, skips  # bottleneck, [skip0, skip1, skip2, skip3]


class Decoder(nn.Module):
    """U-Net decoder with transposed convolution upsampling.

    Each step: UpConv → concat(skip) → ConvBlock
    """

    def __init__(self, out_channels, features=(64, 128, 256, 512), bottleneck=1024, final_activation="tanh"):
        super().__init__()
        self.up_convs = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()

        # Bottleneck → features[-1]
        prev_ch = bottleneck
        reversed_features = list(reversed(features))
        for feat in reversed_features:
            self.up_convs.append(nn.ConvTranspose2d(prev_ch, feat, kernel_size=2, stride=2))
            # Input to conv block: feat (from up) + feat (from skip) = 2*feat
            self.conv_blocks.append(ConvBlock(feat * 2, feat))
            prev_ch = feat

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        if final_activation == "tanh":
            self.final_activation = nn.Tanh()
        elif final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()

    def forward(self, bottleneck, skips):
        """Decode from bottleneck through skip connections."""
        x = bottleneck
        skips_reversed = list(reversed(skips))
        for up_conv, conv_block, skip in zip(self.up_convs, self.conv_blocks, skips_reversed):
            x = up_conv(x)
            # Handle off-by-one size mismatch from pooling
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = conv_block(x)
        x = self.final_conv(x)
        return self.final_activation(x)


class MultiTaskUNet(nn.Module):
    """Shared encoder + dual decoder for NCCT → CTA + vessel segmentation.

    Input:  (B, 1, H, W) NCCT slice
    Output: (B, 1, H, W) CTA prediction (tanh), (B, 1, H, W) vessel seg logits
    """

    def __init__(self, in_channels=1, features=(64, 128, 256, 512), bottleneck=1024):
        super().__init__()
        self.encoder = Encoder(in_channels, features)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], bottleneck, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
        )
        self.cta_decoder = Decoder(1, features, bottleneck, final_activation="tanh")
        # Segmentation decoder outputs logits (no sigmoid) — BCEWithLogitsLoss / DiceLoss expects logits
        self.seg_decoder = Decoder(1, features, bottleneck, final_activation="none")

    def forward(self, x):
        enc_out, skips = self.encoder(x)
        bottleneck = self.bottleneck(enc_out)
        cta_out = self.cta_decoder(bottleneck, skips)
        seg_out = self.seg_decoder(bottleneck, skips)  # logits
        return cta_out, seg_out
