import torch.nn as nn

'''
判别器
'''
class NetD(nn.Module):
    def __init__(self, nz, ndf):
        super().__init__()
        self.ndf = ndf

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=ndf,
            kernel_size=5,
            stride=3,
            padding=1,
            bias=False
        )

        self.conv2 = nn.Conv2d(
            in_channels=ndf,
            out_channels=ndf*2,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

        self.conv3 = nn.Conv2d(
            in_channels=ndf*2,
            out_channels=ndf*4,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

        self.conv4 = nn.Conv2d(
            in_channels=ndf*4,
            out_channels=ndf*8,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

        self.conv5 = nn.Conv2d(
            in_channels=ndf*8,
            out_channels=1,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, input):
        ndf = self.ndf

        x = self.conv1(input)               # 输入：(1, 3, 96, 96)           输出：(1, ngf, 32, 32)
        x = nn.LeakyReLU(0.2, inplace=True)

        x = self.conv2(x)                   # 输入：(1, ngf, 32, 32)         输出：(1, ngf*2, 16, 16)
        x = nn.BatchNorm2d(ndf * 2)
        x = nn.LeakyReLU(0.2, inplace=True)

        x = self.conv3(x)                   # 输入：(1, ngf*2, 16, 16)       输出：(1, ngf*4, 8, 8)
        x = nn.BatchNorm2d(ndf * 4)
        x = nn.LeakyReLU(0.2, inplace=True)

        x = self.conv4(x)                   # 输入：(1, ngf*4, 8, 8)         输出：(1, ngf*8, 4, 4)
        x = nn.BatchNorm2d(ndf * 8)
        x = nn.LeakyReLU(0.2, inplace=True)

        x = self.conv5(x)                       
        return x.view(-1)
