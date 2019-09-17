import torch


def if_but_func(minibatch, sequential, dtype, device):
    if sequential:
        x = [[1] if 'but' in x else [0] for x in minibatch]
        x = torch.tensor(x, dtype=dtype, device=device)
    else:
        raise AttributeError('But rule does not support non-sequential data yet!')
    return x


def but_preprocessing(minibatch):
    if isinstance(minibatch, tuple):
        arr, length = minibatch
    else:
        arr = minibatch
    xx = []
    for x in arr:
        if 'but' not in x:
            xx.append(x)
        else:
            but_idx = x.index('but')
            xx.append(x[but_idx + 1:])
    return xx


but_rule = ('but', if_but_func, but_preprocessing)
