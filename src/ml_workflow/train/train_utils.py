import torch
import torch.optim as optim

def make_optimizer(name_cfg, lr, wd, betas, eps, momentum, params):
    if name_cfg == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
    if name_cfg == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
    if name_cfg == 'sgd':
        return optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum)
    raise ValueError(f"Unsupported optimizer: {name_cfg}")
