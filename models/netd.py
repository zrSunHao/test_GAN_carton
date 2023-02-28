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
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=ndf,
            out_channels=ndf*2,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.norm2 = nn.BatchNorm2d(ndf * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(
            in_channels=ndf*2,
            out_channels=ndf*4,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.norm3 = nn.BatchNorm2d(ndf * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(
            in_channels=ndf*4,
            out_channels=ndf*8,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.norm4 = nn.BatchNorm2d(ndf * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

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

        # 输入：(batch_size, 3, 96, 96)           输出：(batch_size, ngf, 32, 32)
        x = self.conv1(input)
        x = self.relu1(x)

        # 输入：(batch_size, ngf, 32, 32)         输出：(batch_size, ngf*2, 16, 16)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        # 输入：(batch_size, ngf*2, 16, 16)       输出：(batch_size, ngf*4, 8, 8)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        # 输入：(batch_size, ngf*4, 8, 8)         输出：(batch_size, ngf*8, 4, 4)
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        return x.view(-1)
