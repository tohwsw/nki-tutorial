"""step8_softmax_tiled.py — Full tiled softmax with sequential passes."""
import os
os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn2'

import math
import nki
import nki.isa as nisa
import nki.language as nl
import torch


@nki.jit
def nki_softmax_kernel(a):
    num_rows, hidden_dim = a.shape
    TILE = nl.tile_size.pmax
    HIDDEN_TILE = 128
    num_row_tiles = math.ceil(num_rows / TILE)
    num_hidden_tiles = math.ceil(hidden_dim / HIDDEN_TILE)

    out = nl.ndarray((num_rows, hidden_dim), dtype=a.dtype, buffer=nl.shared_hbm)

    for row_idx in nl.affine_range(num_row_tiles):
        row_start = row_idx * TILE
        row_end = min(num_rows, row_start + TILE)
        tile_h = row_end - row_start

        # Initialize row_max to -inf
        row_max = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=row_max, value=-65504.0)

        # Pass 1: find global row max
        for h_idx in nl.sequential_range(num_hidden_tiles):
            h_start = h_idx * HIDDEN_TILE
            h_end = min(hidden_dim, h_start + HIDDEN_TILE)
            h_size = h_end - h_start

            tile = nl.ndarray((TILE, HIDDEN_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tile[0:tile_h, 0:h_size], src=a[row_start:row_end, h_start:h_end])

            local_max = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=local_max, data=tile, op=nl.maximum, axis=(1,))

            nisa.tensor_tensor(dst=row_max, data1=row_max, data2=local_max, op=nl.maximum)

        # Negate row_max for bias
        neg_row_max = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg_row_max, data=row_max, op0=nl.multiply, operand0=-1.0)

        # Pass 2: compute exp(x - max) and accumulate sum
        row_sum = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=row_sum, value=0.0)

        for h_idx in nl.sequential_range(num_hidden_tiles):
            h_start = h_idx * HIDDEN_TILE
            h_end = min(hidden_dim, h_start + HIDDEN_TILE)
            h_size = h_end - h_start

            tile = nl.ndarray((TILE, HIDDEN_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tile[0:tile_h, 0:h_size], src=a[row_start:row_end, h_start:h_end])

            exp_tile = nl.ndarray((TILE, HIDDEN_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=exp_tile, op=nl.exp, data=tile, bias=neg_row_max, scale=1.0)

            local_sum = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=local_sum, data=exp_tile, op=nl.add, axis=(1,))

            nisa.tensor_tensor(dst=row_sum, data1=row_sum, data2=local_sum, op=nl.add)

        # combined_bias = -row_max - log(row_sum)
        log_sum = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=log_sum, op=nl.log, data=row_sum, bias=0.0, scale=1.0)

        combined_bias = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=combined_bias, data1=neg_row_max, data2=log_sum, op=nl.subtract)

        # Pass 3: normalize and store
        for h_idx in nl.affine_range(num_hidden_tiles):
            h_start = h_idx * HIDDEN_TILE
            h_end = min(hidden_dim, h_start + HIDDEN_TILE)
            h_size = h_end - h_start

            tile = nl.ndarray((TILE, HIDDEN_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tile[0:tile_h, 0:h_size], src=a[row_start:row_end, h_start:h_end])

            result = nl.ndarray((TILE, HIDDEN_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=result, op=nl.exp, data=tile, bias=combined_bias, scale=1.0)

            nisa.dma_copy(dst=out[row_start:row_end, h_start:h_end], src=result[0:tile_h, 0:h_size])

    return out


if __name__ == "__main__":
    torch.manual_seed(42)
    import torch_xla
    device = torch_xla.device()

    # Test 1: standard shape
    a = torch.randn((256, 512), dtype=torch.bfloat16)
    ref = torch.softmax(a.float(), dim=-1)
    out = nki_softmax_kernel(a.to(device)).cpu().float()
    passed = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    max_diff = float((out - ref).abs().max())
    print(f"Step 8 — Tiled Softmax (256x512): {'PASS' if passed else 'FAIL'}  max_diff={max_diff:.6f}")
    row_sums = out.sum(dim=-1)
    print(f"  Row sums min={row_sums.min():.4f} max={row_sums.max():.4f}")

    # Test 2: larger hidden dim (forces multiple hidden tiles)
    torch.manual_seed(7)
    a2 = torch.randn((128, 1024), dtype=torch.bfloat16)
    ref2 = torch.softmax(a2.float(), dim=-1)
    out2 = nki_softmax_kernel(a2.to(device)).cpu().float()
    passed2 = torch.allclose(out2, ref2, atol=1e-2, rtol=1e-2)
    max_diff2 = float((out2 - ref2).abs().max())
    print(f"Step 8 — Tiled Softmax (128x1024): {'PASS' if passed2 else 'FAIL'}  max_diff={max_diff2:.6f}")
