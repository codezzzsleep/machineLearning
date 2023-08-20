import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def test(model, data, writer=None):
    model.eval()
    print("Final test of the model")
    output = model(data.x, data.edge_index)
    loss_test = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
    acc = accuracy_score(data.y[data.test_mask], output[data.test_mask])
    if writer is not None:
        global_step = len(data) + 1
        writer.add_scalar('Loss/test', loss_test.item(), global_step)
        writer.add_scalar('Acc/test', acc, global_step)
