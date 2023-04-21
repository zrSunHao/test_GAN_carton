# 默认配置

class DefaultConfig(object):
    # 基本的配置
    data_root = './data'                    # 数据集的存放路径
    models_root = './checkpoints'           # 模型存放路径
    imgs_root = './imgs'                    # 生成图像的保存路径
    image_size = 96                         # 图像尺寸
    device = 'cuda'                         # 使用的设备 cuda/cpu

    # 工具配置信息
    vis_use = True                          # 是否使用 Visdom 可视化
    vis_env = 'GAN'                         # Visdom 的 env
    plot_every = 20                         # 每间隔 20 个batch，Visdom 画图一次

    # 训练相关的配置
    max_epoch = 200                         # 最大训练轮次
    cur_epoch = 1                           # 当前训练的轮次，用于中途停止后再次训练时使用
    save_every = 1                          # 每训练多少个 epoch，保存一次模型
    num_workers = 0                         # 多进程加载数据所用的进程数，默认为0，表示不使用多进程
    batch_size = 32                         # 每批次加载图像的数量
    lr_g = 2e-3                             # 生成器的学习率
    lr_d = 2e-4                             # 判别器的学习率
    betal = 0.5                             # Adam 优化器的 betal 参数

    # 模型相关的配置信息
    nz = 100                                # 噪声维度
    ngf = 64                                # 生成器的特征图数
    ndf = 64                                # 判别器的特征图数
    g_every = 2                             # 每 2 个batch 训练一次生成器
    d_every = 1                             # 每 1 个 batch 训练一次判别器
    netg_path = 'netg_20.pth'               # 与训练的生成器模型路径
    netd_path = 'netd_20.pth'                # 预训练的判别器模型路径

    # 测试相关的配置
    gen_img = 'result.png'                  # 保存的测试图片名称
    gen_search_num = 2000                   # 生成的图像数目
    gen_num = 64                            # 挑选评分最好的图片数目
    gen_mean = 0                            # 噪声的均值
    gen_std = 1                             # 噪声的方差

    