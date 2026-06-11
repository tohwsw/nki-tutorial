"""step4_2d_tiling.py — Tile both partition and free dimensions (no compute)."""
import os
os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn2'

import nki
import nki.isa as nisa
import nki.language as nl
import torch


@nki.jit
def nki_2d_tiling_kernel(a):
    M, N = a.shape
    TILE_P = nl.tile_size.pmax
    TILE_F = 512

    out = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)

    for i_p in nl.affine_range(M // TILE_P):
        for i_f in nl.affine_range(N // TILE_F):
            tile = nl.ndarray((TILE_P, TILE_F), dtype=a.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=tile,
                          src=a[i_p * TILE_P:(i_p + 1) * TILE_P,
                                i_f * TILE_F:(i_f + 1) * TILE_F])

            nisa.dma_copy(dst=out[i_p * TILE_P:(i_p + 1) * TILE_P,
                                  i_f * TILE_F:(i_f + 1) * TILE_F],
                          src=tile)

    return out


if __name__ == "__main__":
    torch.manual_seed(42)
    import torch_xla
    device = torch_xla.device()

    a = torch.randn((256, 1024), dtype=torch.bfloat16)
    ref = a.float()

    out = nki_2d_tiling_kernel(a.to(device)).cpu().float()

    passed = torch.allclose(out, ref, atol=0, rtol=0)
    max_diff = float((out - ref).abs().max())
    print(f"Step 4 — 2D Tiling: {'PASS' if passed else 'FAIL'}  max_diff={max_diff:.6f}  (should be exactly 0)")
