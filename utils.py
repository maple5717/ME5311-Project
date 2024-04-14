from torch.nn import functional as F 

def reshape_to_square(x, size=(80, 80)):
    return F.interpolate(x, size=size, mode='bilinear')

def reshape_back(x, size=(101, 161)):
    return F.interpolate(x, size=size, mode='bilinear')