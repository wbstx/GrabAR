import torch
import torch.nn.functional as F
from torch import nn
from global_context_box import ContextBlock2d

class HandNet(nn.Module):
    def __init__(self):
        super(HandNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU()
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.ReLU()
        )

        self.conv10 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.ReLU()
        )

        self.predict = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1)
        )
        self.tail = nn.Conv2d(32, 2, 1)

    def forward(self, x):

        #x = (x - self.mean) / self.std
        #y = (y - self.mean) / self.std

                                                    # (1 * 7 * 512 * 512)
        d_f1 = self.conv1(x)  # (1 * 32 * 512 * 512)
        d_f2 = self.conv2(d_f1)                     # (1 * 64 * 256 * 256)
        d_f3 = self.conv3(d_f2)                     # (1 * 128 * 128 * 128)
        d_f4 = self.conv4(d_f3)                     # (1 * 256 * 64 * 64)
        # d_f5 = self.conv5(d_f4)                     # (1 * 256 * 64 * 64)
        d_f5 = self.conv5(d_f4)
        # d_f6 = self.conv6(d_f5)                     # (1 * 256 * 64 * 64)
        d_f6 = self.conv6(d_f5) 
        # d_f7 = self.conv7(d_f6)                     # (1 * 256 * 64 * 64)
        d_f7 = self.conv7(d_f6)                      # (1 * 128 * 128 * 128)

        d_f8 = nn.functional.interpolate(d_f7+d_f4, scale_factor=2, mode='bilinear')
        d_f8 = self.conv8(d_f8)                # (1 * 128 * 128 * 128)
        d_f9 = nn.functional.interpolate(d_f8+d_f3, scale_factor=2, mode='bilinear')
        d_f9 = self.conv9(d_f9)                # (1 * 64 * 256 * 256)
        d_f10 = nn.functional.interpolate(d_f9+d_f2, scale_factor=2, mode='bilinear')
        d_f10 = self.conv10(d_f10)              # (1 * 32 * 512 * 512)
        pre = self.predict(d_f10+d_f1)              # (1 * 32 * 512 * 512)
        # pre = self.predict(torch.cat((d_f10, d_f1), 1))
        res = self.tail(pre)                        # (1 * 2 * 512 * 512)

        #return F.sigmoid(res)


        return F.log_softmax(res, dim=1)


