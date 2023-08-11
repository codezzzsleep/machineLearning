import numpy as np
import torch
from torch.utils.data import random_split


def train_valid_split(data_set, valid_ratio, seed):
    '''将提供的训练数据集划分为训练集和验证集'''
    # 计算验证集的大小
    valid_set_size = int(valid_ratio * len(data_set))
    # 计算训练集的大小
    train_set_size = len(data_set) - valid_set_size
    # 使用random_split函数进行划分
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size],
                                        generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)
