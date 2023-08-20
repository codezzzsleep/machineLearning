import torch.nn.functional as F


def train(model, data, epoch, optimizer, path, writer=None,
          criterion=None, seed=42, fastmode=False):
    # 将模型设置为训练模式
    model.train()

    # 迭代数据集进行训练
    for batch_idx, (inputs, targets) in enumerate(data):
        # 将输入数据和目标数据发送到设备上
        inputs = inputs.to(device)
        targets = targets.to(device)

        # 清除之前的梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播
        loss.backward()

        # 更新模型参数
        optimizer.step()

        # 打印训练进度
        if not fastmode and batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(data.dataset),
                       100. * batch_idx / len(data), loss.item()))

        # 将损失值写入TensorBoard
        if writer is not None:
            global_step = epoch * len(data) + batch_idx
            writer.add_scalar('Loss/train', loss.item(), global_step)
