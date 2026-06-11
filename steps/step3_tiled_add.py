"""step3_tiled_add.py — Tiled tensor addition for arbitrary row counts."""
import os
os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn2'

import math
import nki
import nki.isa as nisa
import nki.language as nl
import torch


@nki.jit
def nki_add_tensors_tiled_kernel(a, b):
    num_rows, hidden_dim = a.shape
    TILE = nl.tile_size.pmax
    num_tiles = math.ceil(num_rows / TILE)

    out = nl.ndarray((num_rows, hidden_dim), dtype=a.dtype, buffer=nl.shared_hbm)

    for i in nl.affine_range(num_tiles):
        row_start = i * TILE
        row_end = min(num_rows, row_start + TILE)
        tile_h = row_end - row_start

        a_tile = nl.ndarray((TILE, hidden_dim), dtype=a.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_tile[0:tile_h, 0:hidden_dim], src=a[row_start:row_end, 0:hidden_dim])

        b_tile = nl.ndarray((TILE, hidden_dim), dtype=b.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_tile[0:tile_h, 0:hidden_dim], src=b[row_start:row_end, 0:hidden_dim])

        result = nl.ndarray((TILE, hidden_dim), dtype=a.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=result, data1=a_tile, data2=b_tile, op=nl.add)

        nisa.dma_copy(dst=out[row_start:row_end, 0:hidden_dim], src=result[0:tile_h, 0:hidden_dim])

    return out


if __name__ == "__main__":
    torch.manual_seed(42)
    import torch_xla
    device = torch_xla.device()

    # Test 1: clean multiple of 128
    a = torch.randn((512, 512), dtype=torch.bfloat16)
    b = torch.randn((512, 512), dtype=torch.bfloat16)
    ref = a.float() + b.float()
    out = nki_add_tensors_tiled_kernel(a.to(device), b.to(device)).cpu().float()
    passed = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    max_diff = float((out - ref).abs().max())
    print(f"Step 3 — Tiled Add (512x512): {'PASS' if passed else 'FAIL'}  max_diff={max_diff:.6f}")

    # Test 2: non-multiple of 128 (boundary handling)
    torch.manual_seed(99)
    a2 = torch.randn((300, 512), dtype=torch.bfloat16)
    b2 = torch.randn((300, 512), dtype=torch.bfloat16)
    ref2 = a2.float() + b2.float()
    out2 = nki_add_tensors_tiled_kernel(a2.to(device), b2.to(device)).cpu().float()
    passed2 = torch.allclose(out2, ref2, atol=1e-2, rtol=1e-2)
    max_diff2 = float((out2 - ref2).abs().max())
    print(f"Step 3 — Tiled Add (300x512): {'PASS' if passed2 else 'FAIL'}  max_diff={max_diff2:.6f}")
