from torch import nn
import torch


def double_conv(in_c, out_c):
    layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3),
            nn.ReLU(),)
    return layer


def resize_feature(original, target):
    original_size = original.shape[2]
    target_size = target.shape[2]
    delta = original_size - target_size  # 64 - 56
    delta = delta // 2
    return original[:, :, delta: original_size - delta, delta: original_size-delta]


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = double_conv(1, 64)
        self.conv2 = double_conv(64, 128)
        self.conv3 = double_conv(128, 256)
        self.conv4 = double_conv(256, 512)
        self.conv5 = double_conv(512, 1024)

        self.up_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.conv6 = double_conv(1024, 512)
        self.conv7 = double_conv(512, 256)
        self.conv8 = double_conv(256, 128)
        self.conv9 = double_conv(128, 64)

        self.out = nn.Conv2d(64, 2, kernel_size=3, padding=1)


    def forward(self, xb):
        # encoder - 1x1x572x572
        x1 = self.conv1(xb) # (64x568x568)
        x2 = self.max_pool(x1)
        x3 = self.conv2(x2) # (128x280x280)
        x4 = self.max_pool(x3)
        x5 = self.conv3(x4) # (256x136x136)
        x6 = self.max_pool(x5)
        x7 = self.conv4(x6) # (512x64x64)
        x8 = self.max_pool(x7)
        x9 = self.conv5(x8) ## (1024x28x28)

        # upconv -> resize and concat -> conv
        x10 = self.up_conv1(x9)
        x11 = torch.cat((x10, resize_feature(x7, x10)), dim=1)
        x12 =self.conv6(x11)

        x13 = self.up_conv2(x12)
        x14 = torch.cat((x13, resize_feature(x5, x13)), dim=1)
        x15 =self.conv7(x14)

        x16 = self.up_conv3(x15)
        x17 = torch.cat((x16, resize_feature(x3, x16)), dim=1)
        x18 =self.conv8(x17)

        x18 = self.up_conv4(x18)
        x19 = torch.cat((x18, resize_feature(x1, x18)), dim=1)
        x20 =self.conv9(x19)

        out = self.out(x20)
        return out


if __name__ == "__main__":
    xb = torch.randn(1, 1, 572, 572)
    model = UNet()
    out = model(xb)
    print(out.shape)

