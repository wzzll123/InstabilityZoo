import transformer_engine.pytorch as te
import torch
from torch.nn import functional as F
import math
def max_activation_hook(module, input, output):
    module.max_activation = output[0].abs().max()