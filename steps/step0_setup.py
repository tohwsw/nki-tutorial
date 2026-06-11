"""step0_setup.py — Verify your NKI environment is ready."""
import os
os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn2'

import nki
import nki.isa as nisa
import nki.language as nl
import math
import torch
import torch_xla

device = torch_xla.device()
t = torch.zeros(1, dtype=torch.bfloat16).to(device)
del t

print(f"Platform: {os.environ['NEURON_PLATFORM_TARGET_OVERRIDE']}")
print(f"Torch: {torch.__version__}")
print(f"Neuron device: OK")
print(f"\nReady to write kernels!")
