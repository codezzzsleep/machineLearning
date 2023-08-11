import torch
import numpy as np
from utils import same_seed, save_prediction, select_feature, train_valid_split
from dataset import MyDataset
from torch.utils.data import DataLoader
from net import MyNet
from predict import predict
from train import trainer
import optuna
import pandas as pd
import config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = config.get_config()

# 设置 k-fold 中的 k，这里是根据 valid_ratio 设定的
k = int(1 / config['valid_ratio'])

# Set seed for reproducibility
print(config['seed'])
same_seed(config['seed'])

training_data, test_data = pd.read_csv('../../../dataset/COVID-19/covid_train.csv').values, pd.read_csv(
    '../../../dataset/COVID-19/covid_test.csv').values

num_valid_samples = len(training_data) // k
np.random.shuffle(training_data)
valid_scores = []  # 记录 valid_loss


def objective(trial):
    if trial != None:
        print('\nNew trial here')
        # 定义需要调优的超参数空间
        config['learning_rate'] = trial.suggest_float('lr', 1e-6, 1e-3)
        config['batch_size'] = trial.suggest_categorical('batch_size', [128])
        config['k'] = trial.suggest_int('k_feats', 16, 32)
        config['layer'][0] = config['k']

    # 打印所需的超参数
    print(f'''hyper-parameter: 
        optimizer: {config['optim']},
        lr: {config['learning_rate']}, 
        batch_size: {config['batch_size']}, 
        k: {config['k']}, 
        layer: {config['layer']}''')

    global valid_scores
    # 每次 trial 初始化 valid_scores，可以不初始化，通过 trial * k + fold 来访问当前 trial 的 valid_score，
    # 这样可以让 trainer() 保存 trials 中最好的模型参数，但这并不意味着该参数对应的 k-fold validation loss 最低。
    valid_scores = []

    for fold in range(k):
        # Data split
        valid_data = training_data[num_valid_samples * fold:
                                   num_valid_samples * (fold + 1)]
        train_data = np.concatenate((
            training_data[:num_valid_samples * fold],
            training_data[num_valid_samples * (fold + 1):]))

        # Normalization
        if not config['no_normal']:
            train_mean = np.mean(train_data[:, 35:-1], axis=0)  # 前 35 列为 one-hot vector，我并没有对他们做 normalization，可以自行设置
            train_std = np.std(train_data[:, 35:-1], axis=0)
            train_data[:, 35:-1] -= train_mean
            train_data[:, 35:-1] /= train_std
            valid_data[:, 35:-1] -= train_mean
            valid_data[:, 35:-1] /= train_std
            test_data[:, 35:] -= train_mean
            test_data[:, 35:] /= train_std

        x_train, x_valid, x_test, y_train, y_valid = select_feature(train_data, valid_data, test_data,
                                                                    config)

        train_dataset, valid_dataset, test_dataset = MyDataset(x_train, y_train), \
                                                     MyDataset(x_valid, y_valid), \
                                                     MyDataset(x_test)

        # Pytorch data loader loads pytorch dataset into batches.
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

        model = MyNet(config=config, input_dim=x_train.shape[1]).to(
            device)  # put your model and data on the same computation device.
        valid_score = trainer(train_loader, valid_loader, model, config, device, valid_scores)
        valid_scores.append(valid_score)

        if not config['no_k_cross']:
            break

        if valid_score > 2:
            print(f'在第{fold + 1}折上欠拟合')  # 提前终止，减少计算资源
            break

    print(f'valid_scores: {valid_scores}')

    if trial != None:
        return np.average(valid_scores)
    else:
        return x_test, test_loader


AUTO_TUNE_PARAM = False  # Whether to tune parameters automatically

if AUTO_TUNE_PARAM:
    # 使用Optuna库进行超参数搜索
    n_trials = 10  # 设置试验数量
    print(f'AUTO_TUNE_PARAM: {AUTO_TUNE_PARAM}\nn_trials: {n_trials}')
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    # 输出最优的超参数组合和性能指标
    print('Best hyperparameters: {}'.format(study.best_params))
    print('Best performance: {:.4f}'.format(study.best_value))
else:
    # 注意，只有非自动调参时才进行了predict，节省一下计算资源
    print(f'You could set AUTO_TUNE_PARAM True to tune parameters automatically.\nAUTO_TUNE_PARAM: {AUTO_TUNE_PARAM}')
    x_test, test_loader = objective(None)
    model = MyNet(config=config, input_dim=x_test.shape[1]).to(device)
    model.load_state_dict(torch.load(config['save_path']))
    preds = predict(test_loader, model, device)
    save_prediction(preds, 'submission.csv')
