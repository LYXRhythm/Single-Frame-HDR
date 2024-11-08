
import torch
import torch.nn.functional as F
import torch.nn as nn
import pdb
import math
import numpy as np
from typing import Tuple

def l1(fake, expert, weight=1):
    return (fake - expert).abs().mean()*weight

def psnr(fake, expert):
    # pdb.set_trace()
    mse = (fake - expert).pow(2).mean()
    if mse.pow(2) == 0:
        mse += 1e-6
    if torch.max(expert) > 2:
        max_ = 255.
    else:
        max_ = 1.
    return 10 * torch.log10(max_**2 / (mse)) 

def calculate_psnr(fake, expert):
    mse = np.mean((fake - expert)**2)
    if mse == 0:
        return float('inf')
    # return 20 * math.log10(255.0 / math.sqrt(mse))
    return 20 * math.log10(1.0 / math.sqrt(mse))

def calculate_normalized_psnr(fake, expert, norm):
    
    # normalized_psnr = -10*np.log10(np.mean(np.power(fake/norm - expert/norm, 2)))
    # if normalized_psnr == 0:
    #     return float('inf')
    # return normalized_psnr
    mse = torch.mean((fake / norm - expert / norm) ** 2)
    normalized_psnr = -10 * torch.log10(mse)
    if mse.item() == 0:
        return float('inf')
    
    return normalized_psnr.item()

def cos(fake, expert, weight=1):
    return (1 - torch.nn.functional.cosine_similarity(fake, expert, 1)).mean()*weight

