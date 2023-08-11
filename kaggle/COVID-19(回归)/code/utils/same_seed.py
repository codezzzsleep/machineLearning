import torch
import numpy as np


def same_seed(seed):
    '''为了确保随机数生成器的种子固定，以便能够重现相同的随机结果'''
    # 设置CUDA加速时的确定性算法，确保相同的输入得到相同的输出
    torch.backends.cudnn.deterministic = True
    # 禁用自动寻找最优的卷积算法，确保相同的输入得到相同的输出
    torch.backends.cudnn.benchmark = False
    # 设置NumPy随机数生成器的种子
    np.random.seed(seed)
    # 设置PyTorch随机数生成器的种子
    torch.manual_seed(seed)
    # 如果系统支持CUDA，设置所有可用的CUDA设备的随机数生成器种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
