from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np


def select_feat(train_data, valid_data, test_data, config):
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
