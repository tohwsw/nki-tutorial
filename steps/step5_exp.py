"""step5_exp.py — Element-wise exp() using the Scalar Engine."""
import os
os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn2'

import math
import nki
import nki.isa as nisa
import nki.language as nl
import torch


@nki.jit
def nki_exp_kernel(a):
    num_rows, hidden_dim = a.shape
    TILE = nl.tile_size.pmax
    num_tiles = math.ceil(num_rows / TILE)

    out = nl.ndarray((num_rows, hidden_dim), dtype=a.dtype, buffer=nl.shared_hbm)

    for i in nl.affine_range(num_tiles):
        row_start = i * TILE
        row_end = min(num_rows, row_start + TILE)
        tile_h = row_end - row_start

        x = nl.ndarray((TILE, hidden_dim), dtype=a.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x[0:tile_h, 0:hidden_dim], src=a[row_start:row_end, 0:hidden_dim])

        y = nl.ndarray((TILE, hidden_dim), dtype=a.dtype, buffer=nl.sbuf)
        nisa.activation(dst=y, op=nl.exp, data=x, bias=0.0, scale=1.0)

        nisa.dma_copy(dst=out[row_start:row_end, 0:hidden_dim], src=y[0:tile_h, 0:hidden_dim])

    return out


if __name__ == "__main__":
    torch.manual_seed(42)
    import torch_xla
    device = torch_xla.device()

    a = torch.randn((256, 512), dtype=torch.bfloat16)
    a = a.clamp(-5.0, 5.0)  # prevent overflow in exp
    ref = a.float().exp()

    out = nki_exp_kernel(a.to(device)).cpu().float()

    passed = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    max_diff = float((out - ref).abs().max())
    print(f"Step 5 — Exp: {'PASS' if passed else 'FAIL'}  max_diff={max_diff:.6f}")
