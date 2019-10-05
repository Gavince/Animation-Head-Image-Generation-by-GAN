import torch as t
import torch.nn as nn
from tensorboardX import SummaryWriter


class NetG(nn.Module):
    """
    定义一个生成模型，通过输入噪声来产生一张图片
    """

    def __init__(self, opt):
        super(NetG, self).__init__()
        # ngf = self.ngf  # 生成器的特征图数目
        ngf = opt.ngf
        self.main = nn.Sequential(
            # 假定输入为一张1*1*opt.nz维的数据(opt.nz维的向量)
            nn.ConvTranspose2d(opt.nz , ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),

            # 输入一个４*4*ngf*8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # 输入一个8*8*ngf*4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # 输入一个16*16*ngf*2
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            # 输入一个32*32*ngf
            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()
            # 输出一张96*96*3
        )

    def forward(self, x):
        return self.main(x)


class NetD(nn.Module):
    """
    构建一个判别器，相当与一个二分类问题, 生成一个值
    """

    def __init__(self, opt):
        super(NetD, self).__init__()

        ndf = opt.ndf
        self.main = nn.Sequential(
            # 输入96*96*3
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 输入32*32*ndf
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),

            # 输入16*16*ndf*2
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),

            # 输入为8*8*ndf*4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),

            # 输入为4*4*ndf*8
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True),
            nn.Sigmoid()  # 分类问题
        )

    def forward(self, x):
        return self.main(x).view(-1)


if __name__ == "__main__":
    from config import opt
    input_g = t.Tensor(1, 100, 1, 1)
    input_d = t.Tensor(1, 3, 96, 96)
    netd, netg = NetD(opt), NetG(opt)

    with SummaryWriter(comment="Gan") as w:
        w.add_graph(netg, (input_g,))
        # w.add_graph(netd, (input_d,))
