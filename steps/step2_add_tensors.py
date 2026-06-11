"""step2_add_tensors.py — Add two tensors element-wise."""
import os
os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn2'

import nki
import nki.isa as nisa
import nki.language as nl
import torch


@nki.jit
def nki_add_tensors_kernel(a, b):
    num_rows, hidden_dim = a.shape

    out = nl.ndarray((num_rows, hidden_dim), dtype=a.dtype, buffer=nl.shared_hbm)

    a_tile = nl.ndarray((num_rows, hidden_dim), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_tile, src=a[0:num_rows, 0:hidden_dim])

    b_tile = nl.ndarray((num_rows, hidden_dim), dtype=b.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_tile, src=b[0:num_rows, 0:hidden_dim])

    result = nl.ndarray((num_rows, hidden_dim), dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=result, data1=a_tile, data2=b_tile, op=nl.add)

    nisa.dma_copy(dst=out[0:num_rows, 0:hidden_dim], src=result)

    return out


if __name__ == "__main__":
    torch.manual_seed(42)
    import torch_xla
    device = torch_xla.device()

    a = torch.randn((128, 512), dtype=torch.bfloat16)
    b = torch.randn((128, 512), dtype=torch.bfloat16)
    ref = a.float() + b.float()

    out = nki_add_tensors_kernel(a.to(device), b.to(device)).cpu().float()

    passed = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    max_diff = float((out - ref).abs().max())
    print(f"Step 2 — Add Tensors: {'PASS' if passed else 'FAIL'}  max_diff={max_diff:.6f}")
