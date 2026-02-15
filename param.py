import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from models.WDMFNet import BaseNet

model = BaseNet().cuda()

dummy_input1 = torch.randn(1, 3, 256, 256).cuda()
dummy_input2 = torch.randn(1, 3, 256, 256).cuda()

# FLOPs
flops = FlopCountAnalysis(model, (dummy_input1, dummy_input2))
print("FLOPs:", flops.total() / 1e9, "G") 

# Params
print(parameter_count_table(model))
