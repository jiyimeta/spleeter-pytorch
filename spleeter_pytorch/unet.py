import torch
import torch.nn.functional as F

from torch import nn


class CustomPad(nn.Module):
    def __init__(self, padding_setting=(1, 2, 1, 2)):
        super().__init__()
        self.padding_setting = padding_setting

    def forward(self, x):
        return F.pad(x, self.padding_setting, "constant", 0)


class CustomTransposedPad(nn.Module):
    def __init__(self, padding_setting=(1, 2, 1, 2)):
        super().__init__()
        self.padding_setting = padding_setting

    def forward(self, x):
        l, r, t, b = self.padding_setting
        return x[:, :, l:-r, t:-b]


def get_activation_layer(conv_activation):
    if conv_activation == "LeakyReLU":
        return nn.LeakyReLU(0.2)
    elif conv_activation == "ReLU":
        return nn.ReLU()
    elif conv_activation == "ELU":
        return nn.ELU()
    else:
        raise ValueError(f"Unsupported activation function: {conv_activation}")


def down_block(in_filters, out_filters, conv_activation):
    return nn.Sequential(CustomPad(),
                         nn.Conv2d(in_filters, out_filters, kernel_size=5, stride=2, padding=0)), \
        nn.Sequential(
        nn.BatchNorm2d(out_filters, track_running_stats=True, eps=1e-3, momentum=0.01),
        get_activation_layer(conv_activation))


def up_block(in_filters, out_filters, deconv_activation, dropout=False):
    layers = [
        nn.ConvTranspose2d(in_filters, out_filters, kernel_size=5, stride=2),
        CustomTransposedPad(),
        get_activation_layer(deconv_activation),
        nn.BatchNorm2d(out_filters, track_running_stats=True, eps=1e-3, momentum=0.01)
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))

    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(
            self, in_channels: int, conv_activation: str, deconv_activation: str,
            output_mask_logit: bool):
        super().__init__()
        self.output_mask_logit = output_mask_logit
        self.down1_conv, self.down1_act = down_block(in_channels, 16, conv_activation)
        self.down2_conv, self.down2_act = down_block(16, 32, conv_activation)
        self.down3_conv, self.down3_act = down_block(32, 64, conv_activation)
        self.down4_conv, self.down4_act = down_block(64, 128, conv_activation)
        self.down5_conv, self.down5_act = down_block(128, 256, conv_activation)
        self.down6_conv, self.down6_act = down_block(256, 512, conv_activation)

        self.up1 = up_block(512, 256, deconv_activation, dropout=True)
        self.up2 = up_block(512, 128, deconv_activation, dropout=True)
        self.up3 = up_block(256, 64, deconv_activation, dropout=True)
        self.up4 = up_block(128, 32, deconv_activation)
        self.up5 = up_block(64, 16, deconv_activation)
        self.up6 = up_block(32, 1, deconv_activation)
        self.up7 = nn.Sequential(nn.Conv2d(1, in_channels, kernel_size=4, dilation=2, padding=3))

    def forward(self, x):
        d1_conv = self.down1_conv(x)
        d1 = self.down1_act(d1_conv)

        d2_conv = self.down2_conv(d1)
        d2 = self.down2_act(d2_conv)

        d3_conv = self.down3_conv(d2)
        d3 = self.down3_act(d3_conv)

        d4_conv = self.down4_conv(d3)
        d4 = self.down4_act(d4_conv)

        d5_conv = self.down5_conv(d4)
        d5 = self.down5_act(d5_conv)

        d6_conv = self.down6_conv(d5)
        d6 = self.down6_act(d6_conv)

        u1 = self.up1(d6_conv)
        u2 = self.up2(torch.cat([d5_conv, u1], dim=1))
        u3 = self.up3(torch.cat([d4_conv, u2], dim=1))
        u4 = self.up4(torch.cat([d3_conv, u3], dim=1))
        u5 = self.up5(torch.cat([d2_conv, u4], dim=1))
        u6 = self.up6(torch.cat([d1_conv, u5], dim=1))
        u7 = self.up7(u6)
        if self.output_mask_logit:
            return u7
        else:
            u7 = torch.sigmoid(u7)
            return u7 * x


if __name__ == '__main__':
    net = UNet(14, "LeakyReLU", "ReLU", False)
    print(net(torch.rand(1, 14, 20, 48)).shape)
