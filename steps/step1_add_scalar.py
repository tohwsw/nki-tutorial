"""step1_add_scalar.py — Load a tensor, add 1.0, write back."""
import os
os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn2'

import nki
import nki.isa as nisa
import nki.language as nl
import torch


@nki.jit
def nki_add_scalar_kernel(a):
    num_rows, hidden_dim = a.shape

    out = nl.ndarray((num_rows, hidden_dim), dtype=a.dtype, buffer=nl.shared_hbm)

    x = nl.ndarray((num_rows, hidden_dim), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=a[0:num_rows, 0:hidden_dim])

    y = nl.ndarray((num_rows, hidden_dim), dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=y, data=x, op0=nl.add, operand0=1.0)

    nisa.dma_copy(dst=out[0:num_rows, 0:hidden_dim], src=y)

    return out


if __name__ == "__main__":
    torch.manual_seed(42)
    import torch_xla
    device = torch_xla.device()

    a = torch.randn((128, 512), dtype=torch.bfloat16)
    ref = a.float() + 1.0

    out = nki_add_scalar_kernel(a.to(device)).cpu().float()

    passed = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    max_diff = float((out - ref).abs().max())
    print(f"Step 1 — Add Scalar: {'PASS' if passed else 'FAIL'}  max_diff={max_diff:.6f}")
