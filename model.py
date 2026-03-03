import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class MultiTaskKneeNet(nn.Module):
    def __init__(self, in_ch=2, base_ch=32, num_landmarks=5):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)        # 2 -> 32
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch*2)   # 32 -> 64
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4) # 64 -> 128
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_ch*4, base_ch*8) # 128 -> 256
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_ch*8, base_ch*16) # 256 -> 512

        # Segmentation decoder
        self.up1_seg = UpBlock(base_ch*16, base_ch*8)
        self.up2_seg = UpBlock(base_ch*8, base_ch*4)
        self.up3_seg = UpBlock(base_ch*4, base_ch*2)
        self.up4_seg = UpBlock(base_ch*2, base_ch)
        self.out_seg = nn.Conv2d(base_ch, 1, 1)

        # Landmark decoder
        self.up1_lm = UpBlock(base_ch*16, base_ch*8)
        self.up2_lm = UpBlock(base_ch*8, base_ch*4)
        self.up3_lm = UpBlock(base_ch*4, base_ch*2)
        self.up4_lm = UpBlock(base_ch*2, base_ch)
        self.out_lm = nn.Conv2d(base_ch, num_landmarks, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        # Segmentation decoder
        d1_seg = self.up1_seg(b, e4)
        d2_seg = self.up2_seg(d1_seg, e3)
        d3_seg = self.up3_seg(d2_seg, e2)
        d4_seg = self.up4_seg(d3_seg, e1)
        seg_out = self.out_seg(d4_seg)

        # Landmark decoder
        d1_lm = self.up1_lm(b, e4)
        d2_lm = self.up2_lm(d1_lm, e3)
        d3_lm = self.up3_lm(d2_lm, e2)
        d4_lm = self.up4_lm(d3_lm, e1)
        lm_out = self.out_lm(d4_lm)

        return seg_out, lm_out