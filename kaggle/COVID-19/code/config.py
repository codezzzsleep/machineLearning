def get_config():
    return {
        'seed': 5201314,  # 选择一个随机数种子
        'k': 16,  # Select k features
        'layer': [16, 16],
        'optim': 'SGD',  # 选择优化器
        'momentum': 0.7,
        'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
        'n_epochs': 10000,  # Number of epochs.
        'batch_size': 256,
        'learning_rate': 1e-5,
        'weight_decay': 1e-5,
        'early_stop': 600,  # If model has not improved for this many consecutive epochs, stop training.
        'save_path': './models/model.ckpt',  # Your model will be saved here.
        'no_select_all': True,  # Whether to use all features.
        'no_momentum': True,  # Whether to use momentum
        'no_normal': True,  # Whether to normalize data
        'no_k_cross': False,  # Whether to use K-fold cross validation
        'no_save': False,  # Whether to save model parameters
        'no_tensorboard': False,  # Whether to write tensorboard
    }
