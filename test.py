import torch as t
import os
import torchvision as tv

from configs import DefaultCfg
from models import NetG, NetD

cfg = DefaultCfg()
device = t.device('cuda') if cfg.device == 'cuda' else t.device('cpu')

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
netd.eval()
netg.eval()

with t.no_grad():
    # 定义网络
    noises = t.randn(cfg.gen_search_num, cfg.nz, 1,
                     1).normal_(cfg.gen_mean, cfg.gen_std)
    noises = noises.to(device)

    # 生成图像，并计算图像在判别器中的分数
    fake_imgs = netg(noises)
    scores = netd(fake_imgs).detach()

    # 在生成的图像中挑选最好的几张
    indexs = scores.topk(cfg.gen_num)[1]
    results = []
    for ii in indexs:
        results.append(fake_imgs.data[ii])

    # 保存图像
    img_path = '%s/%s' % (cfg.imgs_root, cfg.gen_img)
    tv.utils.save_image(t.stack(results), img_path,
                        normalize=True, range=(-1, 1))
