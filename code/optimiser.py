import torch

def SGD(model, lr=0.1):
    return torch.optim.SGD(params=model.parameters(), lr=lr)

def Adam(model, lr=0.1):
    return torch.optim.Adam(params = model.parameters(), lr=lr)
