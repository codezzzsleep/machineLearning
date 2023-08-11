import torch
import numpy as np
import csv
from sklearn.feature_selection import SelectKBest, f_regression
from torch.utils.data import random_split


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


def save_prediction(preds, file):
    '''将预测结果保存到指定文件中'''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)  # 创建一个CSV写入器对象
        writer.writerow(['id', 'tested_positive'])  # 写入CSV文件的表头
        for i, p in enumerate(preds):
            writer.writerow([i, p])  # 写入每个预测结果的行，包括id和预测值


def select_feature(train_data, valid_data, test_data, config):
    '''选择对回归任务有用的特征'''
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data

    if not config['no_select_all']:
        # 如果不禁用全选特征，则选择所有特征
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        # 特征选择
        k = config['k']
        selector = SelectKBest(score_func=f_regression, k=k)
        result = selector.fit(train_data[:, :-1], train_data[:, -1])
        idx = np.argsort(result.scores_)[::-1]
        feat_idx = list(np.sort(idx[:k]))

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid


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
