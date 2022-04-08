import imp


import numpy as np
import torch
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True