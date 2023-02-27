# 导入所需的类、函数
import torch as t
from torch.utils.data import DataLoader
import torchvision as tv
from torchnet.meter import AverageValueMeter
import os

from configs import DefaultCfg
from models import NetG, NetD
from tools import Visualizer

# 实例化配置类
cfg = DefaultCfg()
device = t.device('cuda') if cfg.device == 'cuda' else t.device('cpu')
vis = Visualizer(cfg.vis_env)

# 预处理数据，加载数据
transforms = tv.transforms.Compose([
    tv.transforms.Resize(cfg.image_size),
    tv.transforms.CenterCrop(cfg.image_size),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = tv.datasets.ImageFolder(cfg.data_root, transform=transforms)
dataloader = DataLoader(dataset=dataset,
                        batch_size=cfg.batch_size,
                        shuffle=True,
                        num_workers=cfg.num_workers,
                        drop_last=True)

# 实例化模型
netg = NetG(cfg.nz, cfg.ngf)
if not cfg.netg_path is None:
    model_path = os.path.join(cfg.models_root, cfg.netg_path)
    if os.path.exists(model_path):
        state_dict = t.load(model_path)
        netg.load_state_dict(state_dict)
netg.to(device)

netd = NetD(cfg.nz, cfg.ngf)
if not cfg.netd_path is None:
    model_path = os.path.join(cfg.models_root, cfg.netd_path)
    if os.path.exists(model_path):
        state_dict = t.load(model_path)
        netd.load_state_dict(state_dict)
netd.to(device)

# 定义优化器
optimizer_g = t.optim.Adam(
    netg.parameters(), cfg.lr_g, betas=(cfg.betal, 0.999))
optimizer_d = t.optim.Adam(
    netd.parameters(), cfg.lr_d, betas=(cfg.betal, 0.999))

# 定义噪声
# true_labels = t.ones(cfg.batch_size)
# fake_labels = t.zeros(cfg.batch_size)
fix_noises = t.randn(cfg.batch_size, cfg.nz, 1, 1).to(device)

# 训练
errord_meter = AverageValueMeter()
errorg_meter = AverageValueMeter()
epochs = range(cfg.max_epoch)
for epoch in iter(epochs):
    if epoch+1 <= cfg.cur_epoch:
        continue
    for ii, (imgs, _) in enumerate(dataloader):
        real_image = imgs.to(device)

        if ii % cfg.d_every == 0:                           # 训练判别器
            optimizer_d.zero_grad()
            r_preds = netd(real_image)
            noises = t.randn(cfg.batch_size, cfg.nz, 1, 1)
            noises = noises.to(device)
            fake_img = netg(noises)                        # 根据噪声生成假图像
            f_preds = netd(fake_img)
            # 计算损失函数
            r_f_diff = (r_preds - f_preds.mean()).clamp(max=1)
            f_r_diff = (f_preds - r_preds.mean()).clamp(min=-1)
            loss_d_real = (1-r_f_diff).mean()
            loss_d_fake = (1+f_r_diff).mean()
            loss_d = loss_d_real + loss_d_fake
            # 反向传播
            loss_d.backward()
            optimizer_d.step()
            errord_meter.add(loss_d.item())

        if ii % cfg.g_every == 0:                           # 训练生成器
            optimizer_g.zero_grad()
            noises = t.randn(cfg.batch_size, cfg.nz, 1, 1)
            noises = noises.to(device)
            fake_img = netg(noises)
            f_preds = netd(fake_img)
            r_preds = netd(real_image)
            # 计算损失函数
            r_f_diff = r_preds - t.mean(f_preds)
            f_r_diff = f_preds - t.mean(r_preds)
            error_g = t.mean(t.nn.ReLU()(1+r_f_diff)) + \
                t.mean(t.nn.ReLU()(1-f_r_diff))
            # 反向传播
            error_g.backward()
            optimizer_g.step()
            errorg_meter.add(error_g.item())
    
        if cfg.vis_use and ii % cfg.plot_every == cfg.plot_every - 1:
            ## 可视化
            fix_fake_imgs = netg(fix_noises)
            vis.images(fix_fake_imgs.detach().cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
            vis.images(real_image.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
            vis.plot('errord', errord_meter.value()[0])
            vis.plot('errorg', errorg_meter.value()[0])

    if (epoch+1) % cfg.save_every == 0:                     # 保存模型、图片
        fix_fake_imgs = netg(fix_noises)
        img_path = '%s/%s.png' % (cfg.imgs_root, epoch+1)
        tv.utils.save_image(
            fix_fake_imgs[64], img_path, normalize=True, range=(-1, 1))
        t.save(netd.state_dict(), '%s/netd_%s.pth' %
               (cfg.models_root, epoch+1))
        t.save(netg.state_dict(), '%s/netg_%s.pth' %
               (cfg.models_root, epoch+1))
