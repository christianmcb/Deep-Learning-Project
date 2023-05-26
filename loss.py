import torch

#PyTorch
class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
def BCEWithLogitsLoss():
    return torch.nn.BCEWithLogitsLoss()


def IOU(outputs, labels, device):
    output = torch.sigmoid(outputs)
    output = torch.where(output < 0.5, torch.tensor(0.0).to(device), torch.tensor(1.0).to(device)).flatten()
    label = labels.flatten()
    
    intersection = torch.logical_and(label, output)
    union = torch.logical_or(label, output)
    
    return torch.sum(intersection) / torch.sum(union)