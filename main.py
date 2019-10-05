import tqdm
from models import NetD, NetG
from tensorboardX import SummaryWriter
import torch as t
import torchvision as tv
from torch.utils.data import DataLoader
from config import opt
import torch.nn as nn
from torchnet.meter import AverageValueMeter


def train(**kwargs):
    """training NetWork"""

    #  0.配置属性
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = t.device("cuda") if opt.gpu else t.device("cpu")

    # 1.预处理数据
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.img_size),  # 3*96*96
        tv.transforms.CenterCrop(opt.img_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    #  1.1 加载数据
    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)  # TODO 复习这个封装方法
    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.num_workers,
                            drop_last=True)  # TODO 查看drop_last操作

    # 2．初始化网络
    netg, netd = NetG(opt), NetD(opt)
    # 2.1判断网络是否已有权重数值
    map_location = lambda storage, loc: storage  # TODO 复习map_location操作

    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    if opt.netd_path:
        netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    # 2.2 搬移模型到指定设备
    netd.to(device)
    netg.to(device)

    # 3. 定义优化策略
    #  TODO 复习Adam算法
    optimize_g = t.optim.Adam(netg.parameters(), lr=opt.lr1, betas=(opt.beta1,
                                                                    0.999))
    optimize_d = t.optim.Adam(netd.parameters(), lr=opt.lr2, betas=(opt.beta1,
                                                                    0.999))
    criterions = nn.BCELoss().to(device)  # TODO 重新复习BCELoss方法

    # 4. 定义标签, 并且开始注入生成器的输入noise
    true_labels = t.ones(opt.batch_size).to(device)
    fake_labels = t.ones(opt.batch_size).to(device)
    noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    fix_noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)

    errord_meter = AverageValueMeter()  # TODO 重新阅读torchnet
    errorg_meter = AverageValueMeter()

    #  6.训练网络
    epochs = range(opt.max_epoch)
    write = SummaryWriter(log_dir=opt.virs, comment='loss')

    # 6.1 设置迭代
    for epoch in iter(epochs):
        #  6.2 读取每一个batch 数据
        for ii_, (img, _) in tqdm.tqdm(enumerate(dataloader)):
            real_img = img.to(device)

            #  6.3开始训练生成器和判别器
            #  注意要使得生成的训练次数小于一些
            if ii_ % opt.d_every == 0:
                optimize_d.zero_grad()
                # 训练判别器
                # 真图
                output = netd(real_img)
                error_d_real = criterions(output, true_labels)
                error_d_real.backward()

                # 随机生成的假图
                noises = noises.detach()
                fake_image = netg(noises).detach()
                output = netd(fake_image)
                error_d_fake = criterions(output, fake_labels)
                error_d_fake.backward()
                optimize_d.step()

                # 计算loss
                error_d = error_d_fake + error_d_real
                errord_meter.add(error_d.item())

            # 训练判别器
            if ii_ % opt.g_every == 0:
                optimize_g.zero_grad()
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises)
                output = netd(fake_img)
                error_g = criterions(output, true_labels)
                error_g.backward()
                optimize_g.step()

                errorg_meter.add(error_g.item())
            # 绘制数据
            if ii_ % 5 == 0:
                write.add_scalar("Discriminator_loss", errord_meter.value()[0])
                write.add_scalar("Generator_loss", errorg_meter.value()[0])

        #  7.保存模型
        if (epoch + 1) % opt.save_every == 0:
            fix_fake_image = netg(fix_noises)
            tv.utils.save_image(fix_fake_image.data[:64], "%s/%s.png" % (
                opt.save_path, epoch), normalize=True)

            t.save(netd.state_dict(), 'checkpoints/netd_%s.pth' % epoch)
            t.save(netg.state_dict(), 'checkpoints/netg_%s.pth' % epoch)
            errord_meter.reset()
            errorg_meter.reset()

    write.close()


@t.no_grad()
def generate(**kwargs):
    """用训练好的数据进行生成图片"""

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = t.device("cuda" if opt.gpu else "cpu")

    #  1.加载训练好权重数据
    netg, netd = NetG(opt).eval(), NetD(opt).eval()
    map_location = lambda storage, loc: storage

    netd.load_state_dict(t.load(opt.netd_path, map_location=map_location), False)
    netg.load_state_dict(t.load(opt.netg_path, map_location=map_location), False)
    netd.to(device)
    netg.to(device)

    #  2.生成训练好的图片
    noise = t.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean,
                                                              opt.gen_std)
    noise.to(device)

    fake_image = netg(noise)
    score = netd(fake_image).detach()  # TODO 查阅topk()函数

    # 挑选出合适的图片
    indexs = score.topk(opt.gen_num)[1]

    result = []

    for ii in indexs:
        result.append(fake_image.data[ii])

    tv.utils.save_image(t.stack(result), opt.gen_img, normalize=True, range=(-1, 1))


if __name__ == "__main__":
    import fire

    fire.Fire()
