import numpy as np


class EarlyStop:
    def __init__(self, patience=5, delta=0):
        self.patience = patience  # 忍耐的轮数
        self.delta = delta  # 性能变化的阈值
        self.best_loss = np.Inf  # 最佳性能指标
        self.counter = 0  # 没有改善的轮数

    def should_stop(self, current_loss):
        if current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
