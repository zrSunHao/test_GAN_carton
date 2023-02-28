import torch.nn as nn


class NetG(nn.Module):

    '''
    @params nz:  输入噪声的维度
    @params ngf: 生成器的特征图数
    '''
    def __init__(self, nz, ngf):
        super().__init__()
        self.ngf = ngf
        # 输入 nz 维度的噪声，可以看作是一个 1*1*nz 的特征图
        self.convT1 = nn.ConvTranspose2d(in_channels=nz,
                                         out_channels=ngf * 8,
                                         kernel_size=4,
                                         stride=1,
                                         padding=0,
                                         bias=False)
        self.norm1 = nn.BatchNorm2d(ngf * 8)
        self.relu1 = nn.ReLU(True)

        self.convT2 = nn.ConvTranspose2d(in_channels=ngf * 8,
                                         out_channels=ngf * 4,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1,
                                         bias=False)
        self.norm2 = nn.BatchNorm2d(ngf * 4)
        self.relu2 = nn.ReLU(True)

        self.convT3 = nn.ConvTranspose2d(in_channels=ngf * 4,
                                         out_channels=ngf * 2,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1,
                                         bias=False)
        self.norm3 = nn.BatchNorm2d(ngf * 2)
        self.relu3 = nn.ReLU(True)

        self.convT4 = nn.ConvTranspose2d(in_channels=ngf * 2,
                                         out_channels=ngf,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1,
                                         bias=False)
        self.norm4 = nn.BatchNorm2d(ngf)
        self.relu4 = nn.ReLU(True)

        self.convT5 = nn.ConvTranspose2d(in_channels=ngf,
                                         out_channels=3,
                                         kernel_size=5,
                                         stride=3,
                                         padding=1,
                                         bias=False)
        self.tanh5 = nn.Tanh()

    def forward(self,input):
        ngf = self.ngf

        x = self.convT1(input)          # 输入：(batch_size, nz, 1, 1)        输出：(batch_size, ngf*8, 4, 4)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.convT2(x)              # 输入：(batch_size, ngf*8, 4, 4)     输出：(batch_size, ngf*4, 8, 8)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.convT3(x)              # 输入：(batch_size, ngf*4, 8, 8)     输出：(batch_size, ngf*2, 16, 16)
        x = self.norm3(x)
        x = self.relu3(x)

        x = self.convT4(x)              # 输入：(batch_size, ngf*2, 16, 16)   输出：(batch_size, ngf, 32, 32)
        x = self.norm4(x)
        x = self.relu4(x)

        x = self.convT5(x)              # 输入：(1, ngf, 32, 32)   输出：(1, 3, 96, 96)
        x = self.tanh5(x)

        return x

