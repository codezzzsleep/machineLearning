from torch.utils.tensorboard import SummaryWriter

write = SummaryWriter()

for i in range(100):
    write.add_scalar(tag="y=x", scalar_value=i, global_step=i)

write.close()
