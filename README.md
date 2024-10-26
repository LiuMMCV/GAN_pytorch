# 示例工程，利用GAN生成手写数字

## 训练

```
git clone https://github.com/LiuMMCV/GAN_pytorch.git
cd GAN_pytorch
pip install -r requirements.txt
python main.py
```

## 计算推理时间

```
python event.py
```

## 服务器环境配置

cuda11.8+cudnn8.6+python3.8

### 安装pytorch

```
#进入虚拟环境
conda activate [你的虚拟环境名]

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

测试GPU
############
#进入虚拟环境
conda activate [你的虚拟环境名]

#输入python来进入python的环境
python

#加载torch
import torch

print(torch.backends.cudnn.version())
#输出8200，代表着成功安装了cudnn v8.4.0

print(torch.__version__)
#输出1.11.0，代表成功安装了pytorch 1.11.0

print(torch.version.cuda)
#输出11.8，代表成功安装了cuda 11.8

torch.cuda.is_available()
#True
###########
```

### 



