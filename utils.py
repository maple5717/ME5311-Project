from torch.nn import functional as F 
from config import *

def reshape_to_square(x, size=reshape_size):
    return F.interpolate(x, size=size, mode='bilinear')

def reshape_back(x, size=(101, 161)):
    return F.interpolate(x, size=size, mode='bilinear')