# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from .pscc import PSCC as MODEL
from .mseloss import MSELoss

def build_model(config):
    model = MODEL()
    return model, MSELoss(config.FACTOR)
