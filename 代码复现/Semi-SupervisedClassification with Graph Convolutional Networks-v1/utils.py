import os
import time
from pathlib import Path
import torch
import numpy as np


def create_result_folder():
    # 创建结果文件夹
    result_folder = f"{time.strftime('%Y%m%d_%H%M%S')}"
    result_folder = os.path.join("result", result_folder)
    train_folder = os.path.join(result_folder, 'train')
    test_folder = os.path.join(result_folder, 'test')
    Path(train_folder).mkdir(parents=True, exist_ok=True)
    Path(test_folder).mkdir(parents=True, exist_ok=True)
    return train_folder, test_folder


# 示例：将一些结果保存到结果文件夹中
def save_result(result_folder, is_train=True):
    if is_train:
        folder = result_folder[0]  # 使用训练集文件夹路径
    else:
        folder = result_folder[1]  # 使用测试集文件夹路径
    result_file = os.path.join(folder, 'result.txt')
    with open(result_file, 'w') as f:
        f.write('This is a sample result.')
    print(f"结果已保存到文件夹：{folder}")


# 调用示例
result_folders = create_result_folder()
save_result(result_folders, is_train=True)
save_result(result_folders, is_train=False)


def get_current_path():
    return os.getcwd()


def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    same_seed(42)
