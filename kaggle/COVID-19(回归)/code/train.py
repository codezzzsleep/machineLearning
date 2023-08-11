import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import math
import os


def trainer(train_loader, valid_loader, model, config, device, valid_scores):
    # Define your loss function, do not modify this.
    criterion = nn.MSELoss(reduction='mean')

    # Define your optimization algorithm.
    if config['optim'] == 'SGD':
        if config['no_momentum']:
            optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'],
                                        weight_decay=config['weight_decay'])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'],
                                        weight_decay=config['weight_decay'])
    elif config['optim'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                     weight_decay=config['weight_decay'])

    # Writer of tensoboard.
    writer = SummaryWriter()

    if not os.path.isdir('./models'):
        os.mkdir('./models')  # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()  # Set your model to train mode.
        loss_record = []

        # 如果你在kaggle上运行，可以注释掉大部分的打印函数，并将train_pbar注释掉，令 x,y in train_loader，因为kaggle上打印太多可能会报错。
        # tqdm is a package to visualize your training progress.
        # train_pbar = tqdm(train_loader, position=0, leave=True)
        # for x, y in train_pbar:
        for x, y in train_loader:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            # train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            # train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)

        model.eval()  # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)

        # if epoch % 100 == 0:
        #    print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')

        if not config['no_tensorboard']:
            writer.add_scalar('Loss/train', mean_train_loss, step)
            writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss

            # 一轮实验中保存 K 折交叉验证中单折表现最好的模型
            if len(valid_scores):
                if best_loss < min(valid_scores):
                    torch.save(model.state_dict(), config['save_path'])  # Save your best model
                    # print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
                    print('Saving model with loss {:.3f}...'.format(best_loss))
            else:
                torch.save(model.state_dict(), config['save_path'])  # Save your best model
                # print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
                print('Saving model with loss {:.3f}...'.format(best_loss))

            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('Best loss {:.3f}...'.format(best_loss))
            print('\nModel is not improving, so we halt the training session.')
            break
    return best_loss
