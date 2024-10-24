# import progressbar
# from tqdm import tqdm
# import time
#
# # //定义进度条的显示样式
# widgets = ["doing task: ", progressbar.Percentage()," ",
#         progressbar.Bar(), " ", progressbar.ETA()]
#
# # //创建进度条并开始运行
# pbar = progressbar.ProgressBar(maxval=100, widgets=widgets).start()
#
# for i in tqdm(range(100)):
#     time.sleep(0.1)
#     # //更新进度
#     pbar.update(i)
#
# # //结束进度条
# pbar.finish()
#
from tqdm import tqdm
import time

for i in tqdm(range(100)):
    time.sleep(0.5)
