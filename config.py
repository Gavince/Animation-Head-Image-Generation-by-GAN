class Config(object):
    """
    定义一个配置类
    """
    # 0.参数调整
    data_path = "data/"
    virs = "result"
    num_workers = 4  # 多线程
    img_size = 96  # 剪切图片的大小
    batch_size = 256
    max_epoch = 200
    lr1 = 2e-4  # 生成器adam
    lr2 = 2e-4  # 判别器Adam
    beta1 = 0.5
    gpu = True
    nz = 100  # 噪声维度
    ngf = 64  # 生成器的feature map 数
    ndf = 64  # 判别器的feature map 数

    # 1.模型保存路径
    save_path = 'imgs/'  # 生成图片的保存路径
    d_every = 1  # 每一个batch 训练一次判别器
    g_every = 5  # 每5个batch训练一次生成模型
    save_every = 10  # 每10次保存一次模型
    netd_path = None
    netg_path = None

    # 2.测试数据
    gen_img = "result.png"
    # 选择保存的照片
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0
    gen_std = 1


opt = Config()