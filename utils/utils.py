from turtle import forward
import numpy as np
import torch
from torch import nn, zero_
import time


def save_checkpoint(dict, path="model.pth.tar"):
    """
    save the model and optim through a dictionary
     {"model": model.state_dict(), "optim": optim.state_dict()} (*)

    params:
        data: a dict of the structure (*) storing the model and optim

    return:
        None
    """
    now = time.strftime("%D_%H:%M")
    print(f"saving checkpoint at {now}, path is '{path}'")
    torch.save(dict, path)
    print("saving model... done!")


def set_seed(seed):
    """
    set random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def init_weight(m):
    """
    init model's weights
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    if isinstance(m, (nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




class BasicConv(nn.Module):
    """
    basic conv block has structure: Conv - bn - relu
    you can choose use bn or not
    you can choose use relu or leakyrelu
    """
    def __init__(self, in_channel, out_channel, k_s, s, bn=False, leaky_relu=True) -> None:
        super(BasicConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, k_s, s, k_s//2),
            nn.Identity() if bn==False else nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2) if leaky_relu==True else nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)