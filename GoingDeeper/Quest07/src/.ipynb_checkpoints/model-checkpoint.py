import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    # 호환성을 위해 deep_supervision 인자를 받지만 사용하지는 않음 (Dummy Argument)
    def __init__(self, input_channels=3, output_channels=5, deep_supervision=False):
        super(UNet, self).__init__()

        # Contracting Path (Encoder)
        self.enc1 = self.double_conv(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self.double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.double_conv(512, 1024)
        self.dropout = nn.Dropout(0.5)

        # Expanding Path (Decoder)
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec6 = self.double_conv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec7 = self.double_conv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec8 = self.double_conv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec9 = self.double_conv(128, 64)

        # Output layer
        self.final = nn.Conv2d(64, output_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        """2개의 Conv Layer로 이루어진 블록"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        c1 = self.enc1(x)
        p1 = self.pool1(c1)
        c2 = self.enc2(p1)
        p2 = self.pool2(c2)
        c3 = self.enc3(p2)
        p3 = self.pool3(c3)
        c4 = self.enc4(p3)
        p4 = self.pool4(c4)

        # Bottleneck
        c5 = self.bottleneck(p4)
        c5 = self.dropout(c5)

        # Decoder
        u6 = self.up6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.dec6(u6)

        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.dec7(u7)

        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.dec8(u8)

        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.dec9(u9)

        # Output
        output = self.final(c9)
        return output

class ConvBlockNested(nn.Module):
    """
    U-Net++의 각 노드에서 사용되는 Convolution Block
    (Conv3x3 -> BN -> ReLU) x 2
    """
    def __init__(self, in_ch, mid_ch, out_ch):
        super(ConvBlockNested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        return x

class NestedUNet(nn.Module):
    """
    [U-Net++]
    Paper: UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation
    """
    def __init__(self, input_channels=3, output_channels=5, deep_supervision=False):
        super(NestedUNet, self).__init__()

        nb_filter = [64, 128, 256, 512, 1024]
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # -----------------------------------------------------------------
        # Encoder (Backbone)
        # -----------------------------------------------------------------
        self.conv0_0 = ConvBlockNested(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = ConvBlockNested(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = ConvBlockNested(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = ConvBlockNested(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = ConvBlockNested(nb_filter[3], nb_filter[4], nb_filter[4])

        # -----------------------------------------------------------------
        # Nested Skip Pathways (Dense Connections)
        # -----------------------------------------------------------------
        # X_0_j
        self.conv0_1 = ConvBlockNested(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_2 = ConvBlockNested(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_3 = ConvBlockNested(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_4 = ConvBlockNested(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        # X_1_j
        self.conv1_1 = ConvBlockNested(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv1_2 = ConvBlockNested(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv1_3 = ConvBlockNested(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        # X_2_j
        self.conv2_1 = ConvBlockNested(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv2_2 = ConvBlockNested(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        # X_3_j
        self.conv3_1 = ConvBlockNested(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        # -----------------------------------------------------------------
        # Output Layers (Deep Supervision)
        # -----------------------------------------------------------------
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)

    def forward(self, x):
        # Row 0
        x0_0 = self.conv0_0(x)

        # Row 1
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        # Row 2
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        # Row 3
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        # Row 4
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            # Deep Supervision이 켜져있을 경우 모든 Scale의 출력을 반환
            return [output1, output2, output3, output4]
        else:
            # 기본적으로는 마지막 레이어의 출력만 반환 (기존 Training Loop 호환)
            return self.final(x0_4)
