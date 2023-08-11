def get_config():
    '''获取配置字典'''
    return {
        'seed': 5201314,  # 选择一个随机数种子
        'k': 16,  # 选择的特征数量
        'layer': [16, 16],  # 神经网络的层结构
        'optim': 'SGD',  # 选择优化器
        'momentum': 0.7,  # 动量参数
        'valid_ratio': 0.2,  # 验证集占训练集的比例
        'n_epochs': 10000,  # 迭代的轮数
        'batch_size': 256,  # 批量大小
        'learning_rate': 1e-5,  # 学习率
        'weight_decay': 1e-5,  # 权重衰减参数
        'early_stop': 600,  # 如果模型连续多少轮未改善，则停止训练
        'save_path': '../models/model.ckpt',  # 模型保存路径
        'no_select_all': True,  # 是否使用全部特征
        'no_momentum': True,  # 是否使用动量
        'no_normal': True,  # 是否对数据进行归一化
        'no_k_cross': False,  # 是否使用K折交叉验证
        'no_save': False,  # 是否保存模型参数
        'no_tensorboard': False,  # 是否记录TensorBoard日志
    }
