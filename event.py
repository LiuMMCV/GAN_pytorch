import argparse
import numpy as np
import torch.nn as nn
import torch
from thop import profile, clever_format
from torchsummary import summary


# 将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
# 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类
class Discriminator(nn.Module):
    def __init__(self, channels, img_size):
        super(Discriminator, self).__init__()
        img_shape = (channels, img_size, img_size)
        img_area = np.prod(img_shape)

        self.model = nn.Sequential(
            nn.Linear(img_area, 512),  # 输入特征数为784，输出为512
            nn.LeakyReLU(0.2, inplace=True),  # 进行非线性映射
            nn.Linear(512, 256),  # 输入特征数为512，输出为256
            nn.LeakyReLU(0.2, inplace=True),  # 进行非线性映射
            nn.Linear(256, 1),  # 输入特征数为256，输出为1
            nn.Sigmoid(),  # sigmoid是一个激活函数，二分类问题中可将实数映射到[0, 1],作为概率值, 多分类用softmax函数
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)  # 鉴别器输入是一个被view展开的(784)的一维图像:(64, 784)
        validity = self.model(img_flat)  # 通过鉴别器网络
        return validity  # 鉴别器返回的是一个[0, 1]间的概率


# 输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维,
# 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，
# 然后经过线性变换将其变成784维，最后经过Tanh激活函数是希望生成的假的图片数据分布, 能够在-1～1之间。
class Generator(nn.Module):
    def __init__(self, channels, img_size, latent_dim):
        super(Generator, self).__init__()
        self.img_shape = (channels, img_size, img_size)
        img_area = np.prod(self.img_shape)

        # 模型中间块儿
        def block(in_feat, out_feat, normalize=True):  # block(in， out )
            layers = [nn.Linear(in_feat, out_feat)]  # 线性变换将输入映射到out维
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))  # 正则化
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # 非线性激活函数
            return layers

        # prod():返回给定轴上的数组元素的乘积:1*28*28=784
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),  # 线性变化将输入映射 100 to 128, 正则化, LeakyReLU
            *block(128, 256),  # 线性变化将输入映射 128 to 256, 正则化, LeakyReLU
            *block(256, 512),  # 线性变化将输入映射 256 to 512, 正则化, LeakyReLU
            *block(512, 1024),  # 线性变化将输入映射 512 to 1024, 正则化, LeakyReLU
            nn.Linear(1024, img_area),  # 线性变化将输入映射 1024 to 784
            nn.Tanh()  # 将(784)的数据每一个都映射到[-1, 1]之间
        )

    # view():相当于numpy中的reshape，重新定义矩阵的形状:这里是reshape(64, 1, 28, 28)
    def forward(self, z):  # 输入的是(64， 100)的噪声数据
        imgs = self.model(z)  # 噪声数据通过生成器模型
        imgs = imgs.view(imgs.size(0), *self.img_shape)  # reshape成(64, 1, 28, 28)
        return imgs  # 输出为64张大小为(1, 28, 28)的图像


def cal_time(model, x):
    with torch.inference_mode():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        time_list = []
        for _ in range(50):
            start_event.record()
            ret = model(x)
            end_event.record()
            end_event.synchronize()
            time_list.append(start_event.elapsed_time(end_event) / 1000)

        print(f"event avg time: {sum(time_list[5:]) / len(time_list[5:]):.5f} s")
        print(f"FPS: {len(time_list[5:]) / sum(time_list[5:]):.5f}")


def main(opt):
    device = torch.device('cuda:0' if opt.n_gpu > 0 else 'cpu')

    generator = Generator(channels=opt.channels, img_size=opt.img_size, latent_dim=opt.latent_dim).to(device)
    discriminator = Discriminator(channels=opt.channels, img_size=opt.img_size).to(device)

    ckpt_g = torch.load('./save/gan/generator.pth')
    ckpt_d = torch.load('./save/gan/discriminator.pth')
    generator.load_state_dict(ckpt_g)
    discriminator.load_state_dict(ckpt_d)

    # 冻结权重
    # for name, param in generator.model.named_parameters():
    #     if "0" in name:
    #         param.requires_grad = False
    #
    # print("model.fc1.weight", generator.model[0].weight)
    # print("model.fc2.weight", discriminator.model[0].weight)

    print("========discriminator=========")
    # 模拟输入
    x = torch.randn(size=(opt.batch_size, opt.channels, opt.img_size, opt.img_size), device=device)
    # 打印网络结构
    summary(discriminator, input_size=(opt.channels, opt.img_size, opt.img_size))
    # 计算FLOPs和params
    flops, params = profile(discriminator, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")  # 格式化数据
    print("flops:", flops)
    print("params:", params)
    # 计算推理时间
    cal_time(discriminator, x)
    print("========discriminator=========\n\n")

    print("========generator=========")
    # 模拟输入
    y = torch.randn(size=(opt.batch_size, opt.latent_dim), device=device)
    # 打印网络结构
    summary(generator, input_size=(opt.latent_dim,))
    # 计算FLOPs和params
    flops, params = profile(generator, inputs=(y,))
    flops, params = clever_format([flops, params], "%.3f")  # 格式化数据
    print("flops:", flops)
    print("params:", params)
    # 计算推理时间
    cal_time(generator, y)
    print("========generator=========")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval betwen image samples")
    config = parser.parse_args()
    # opt = parser.parse_args(args=[])                 # 在colab中运行时，换为此行
    print(config)
    main(config)
