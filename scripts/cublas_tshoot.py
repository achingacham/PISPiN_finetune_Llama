print("BoC")

import torch
import torch.nn.functional as F
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
#
pad = torch.rand((1), requires_grad=True, device="cuda")
# A = torch.rand((5120, 2560), requires_grad=True, dtype=torch.half, device="cuda")
print("A formed")
# all_tensors = [pad, A]
# new_tensors = _unflatten_dense_tensors(_flatten_dense_tensors([p.clone().detach() for p in all_tensors]), all_tensors)
# pad, A = new_tensors
#
# print("new_tensor formed")
#
# X = torch.rand((26, 1, 2560), requires_grad=True, dtype=torch.half, device="cuda")
# B = torch.rand((5120), requires_grad=True, dtype=torch.half, device="cuda")
# print("X formed")
#
# out = F.linear(X, A, B)
# print(out)

print("EoC")
