"""step7_softmax_simple.py — Row-wise softmax (single tile, no loops)."""
import os
os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn2'

import nki
import nki.isa as nisa
import nki.language as nl
import torch


@nki.jit
def nki_softmax_simple_kernel(a):
    num_rows, hidden_dim = a.shape

    out = nl.ndarray((num_rows, hidden_dim), dtype=a.dtype, buffer=nl.shared_hbm)

    x = nl.ndarray((num_rows, hidden_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=a[0:num_rows, 0:hidden_dim])

    # Row max
    row_max = nl.ndarray((num_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=row_max, data=x, op=nl.maximum, axis=(1,))

    # Negate row_max for use as bias
    neg_row_max = nl.ndarray((num_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=neg_row_max, data=row_max, op0=nl.multiply, operand0=-1.0)

    # exp(x - row_max) via activation bias broadcast
    exp_vals = nl.ndarray((num_rows, hidden_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=exp_vals, op=nl.exp, data=x, bias=neg_row_max, scale=1.0)

    # Row sum of exp values
    row_sum = nl.ndarray((num_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=row_sum, data=exp_vals, op=nl.add, axis=(1,))

    # log(row_sum)
    log_sum = nl.ndarray((num_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=log_sum, op=nl.log, data=row_sum, bias=0.0, scale=1.0)

    # combined_bias = -row_max - log(sum)
    combined_bias = nl.ndarray((num_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=combined_bias, data1=neg_row_max, data2=log_sum, op=nl.subtract)

    # softmax = exp(x - row_max - log(sum)) = exp(x-max)/sum
    result = nl.ndarray((num_rows, hidden_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=result, op=nl.exp, data=x, bias=combined_bias, scale=1.0)

    nisa.dma_copy(dst=out[0:num_rows, 0:hidden_dim], src=result)

    return out


if __name__ == "__main__":
    torch.manual_seed(42)
    import torch_xla
    device = torch_xla.device()

    a = torch.randn((128, 512), dtype=torch.bfloat16)
    ref = torch.softmax(a.float(), dim=-1)

    out = nki_softmax_simple_kernel(a.to(device)).cpu().float()

    passed = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    max_diff = float((out - ref).abs().max())
    print(f"Step 7 — Simple Softmax: {'PASS' if passed else 'FAIL'}  max_diff={max_diff:.6f}")

    # Sanity check: rows should sum to 1.0
    row_sums = out.sum(dim=-1)
    print(f"  Row sums — min: {row_sums.min():.4f}  max: {row_sums.max():.4f}  (should be ~1.0)")
