"""step9_fused_matmul.py — Fused matmul + bias + ReLU using the Tensor Engine."""
import os
os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn2'

import nki
import nki.isa as nisa
import nki.language as nl
import torch


@nki.jit
def nki_fused_matmul_bias_relu(AT, B, bias):
    """
    Fused kernel: C = relu(A @ B + bias)

    A is passed pre-transposed as AT [K, M] so that K is on the partition axis,
    as required by nc_matmul (both operands need K on their partition axis).

    Args:
        AT:   shape [K, M] - A transposed. K and M multiples of 128.
        B:    shape [K, N] - K multiple of 128, N multiple of 512.
        bias: shape [1, N] - bias row vector, broadcast across M rows.
    Returns:
        C: shape [M, N], non-negative (ReLU applied).
    """
    K, M = AT.shape
    K2, N = B.shape

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax                   # 128
    TILE_N = nl.tile_size.gemm_moving_fmax       # 512

    assert K == K2, f"K mismatch: AT has K={K}, B has K={K2}"
    assert M % TILE_M == 0 and K % TILE_K == 0 and N % TILE_N == 0

    # Allocate output in HBM
    C = nl.ndarray((M, N), dtype=AT.dtype, buffer=nl.shared_hbm)

    # Outer loops tile the output; inner loop accumulates over K
    for i_m in nl.affine_range(M // TILE_M):
        for i_n in nl.affine_range(N // TILE_N):

            # PSUM accumulator — float32, wired to Tensor Engine output
            acc = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

            for i_k in nl.affine_range(K // TILE_K):
                a_tile = nl.ndarray((TILE_K, TILE_M), dtype=AT.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=a_tile,
                              src=AT[i_k * TILE_K:(i_k + 1) * TILE_K,
                                     i_m * TILE_M:(i_m + 1) * TILE_M])

                b_tile = nl.ndarray((TILE_K, TILE_N), dtype=B.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=b_tile,
                              src=B[i_k * TILE_K:(i_k + 1) * TILE_K,
                                    i_n * TILE_N:(i_n + 1) * TILE_N])

                nisa.nc_matmul(dst=acc, stationary=a_tile, moving=b_tile)

            # Move result from PSUM to SBUF
            matmul_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=AT.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=matmul_sbuf, src=acc)

            # Add bias — broadcast (1, TILE_N) to (TILE_M, TILE_N)
            bias_tile = nl.ndarray((TILE_M, TILE_N), dtype=bias.dtype, buffer=nl.sbuf)
            for i_p in nl.affine_range(TILE_M):
                nisa.dma_copy(dst=bias_tile[i_p:i_p+1, 0:TILE_N],
                              src=bias[0:1, i_n*TILE_N:(i_n+1)*TILE_N])

            biased = nl.ndarray((TILE_M, TILE_N), dtype=AT.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=biased, data1=matmul_sbuf, data2=bias_tile, op=nl.add)

            # Apply ReLU and store
            relu_result = nl.ndarray((TILE_M, TILE_N), dtype=AT.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=relu_result, data=biased, op0=nl.maximum, operand0=0.0)

            nisa.dma_copy(dst=C[i_m*TILE_M:(i_m+1)*TILE_M, i_n*TILE_N:(i_n+1)*TILE_N],
                          src=relu_result)

    return C


if __name__ == "__main__":
    torch.manual_seed(0)
    import torch_xla
    device = torch_xla.device()

    M, K, N = 256, 256, 512
    A = torch.randn((M, K), dtype=torch.float16)
    B = torch.randn((K, N), dtype=torch.float16)
    bias = torch.randn((1, N), dtype=torch.float16)
    AT = A.T.contiguous()  # [K, M] — K on partition axis

    ref = torch.clamp(A.float() @ B.float() + bias.float(), min=0)

    out = nki_fused_matmul_bias_relu(AT.to(device), B.to(device), bias.to(device)).cpu().float()

    passed = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    max_diff = float((out - ref).abs().max())
    print(f"Step 9 — Fused Matmul+Bias+ReLU: {'PASS' if passed else 'FAIL'}  max_diff={max_diff:.6f}")
    print(f"  Output shape: {out.shape}  min={out.min():.4f} (should be >= 0)")
