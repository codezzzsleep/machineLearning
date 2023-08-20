import torch.nn.functional as F
from tqdm import tqdm
import utils
import sys
from sklearn.metrics import classification_report, accuracy_score


def train(model, data, epochs, optimizer, path, writer=None,
          criterion=None, fastmode=False):
    # 将模型设置为训练模式
    best_loss = sys.float_info.max
    model.train()
    print("train start……")
    for epoch in tqdm(range(epochs), desc='Processing'):
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss_train = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        acc_train = accuracy_score(data.y[data.train_mask], output[data.train_mask])
        loss_train.backward()
        optimizer.step()
        # 将损失值写入TensorBoard
        if writer is not None:
            global_step = epoch + 1
            writer.add_scalar('Loss/train', loss_train.item(), global_step)
            writer.add_scalar('Acc/train', acc_train, global_step)
        if fastmode:
            model.eval()
            output = model(data.x, data.edge_index)
            loss_val = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
            acc_val = accuracy_score(data.y[data.val_mask], output[data.val_mask])
            best_loss = utils.update_best_model(model, loss_val.item(), best_loss, path[0])
            if writer is not None:
                global_step = epoch + 1
                writer.add_scalar('Loss/val', loss_val.item(), global_step)
                writer.add_scalar('Acc/val', acc_val, global_step)

    utils.save_result(model, path, True)
