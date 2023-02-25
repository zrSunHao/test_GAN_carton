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

        self.convT2 = nn.ConvTranspose2d(in_channels=ngf * 8,
                                         out_channels=ngf * 4,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1,
                                         bias=False)

        self.convT3 = nn.ConvTranspose2d(in_channels=ngf * 4,
                                         out_channels=ngf * 2,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1,
                                         bias=False)

        self.convT4 = nn.ConvTranspose2d(in_channels=ngf * 2,
                                         out_channels=ngf,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1,
                                         bias=False)

        self.convT5 = nn.ConvTranspose2d(in_channels=ngf,
                                         out_channels=3,
                                         kernel_size=5,
                                         stride=3,
                                         padding=1,
                                         bias=False)
        
    def forward(self,input):
        ngf = self.ngf

        x = self.convT1(input)          # 输入：(1, nz, 1, 1)        输出：(1, ngf*8, 4, 4)
        nn.BatchNorm2d(ngf * 8)
        nn.ReLU(True)

        x = self.convT2(x)              # 输入：(1, ngf*8, 4, 4)     输出：(1, ngf*4, 8, 8)
        nn.BatchNorm2d(ngf * 4)
        nn.ReLU(True)

        x = self.convT2(x)              # 输入：(1, ngf*4, 8, 8)     输出：(1, ngf*2, 16, 16)
        nn.BatchNorm2d(ngf * 2) 
        nn.ReLU(True)

        x = self.convT2(x)              # 输入：(1, ngf*2, 16, 16)   输出：(1, ngf, 32, 32)
        nn.BatchNorm2d(ngf)
        nn.ReLU(True)

        x = self.convT2(x)              # 输入：(1, ngf, 32, 32)   输出：(1, 3, 96, 96)
        nn.Tanh

        return x

