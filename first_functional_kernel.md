# Your First Functional Kernel on AWS Trainium

By the end of this workshop, you will have written a **row-wise softmax kernel** and a **fused matmul+bias+ReLU kernel** from scratch using the NKI ISA-path API, running on real Trainium hardware.

## Why Softmax?

Softmax is the operation that converts a row of raw numbers into a probability distribution — all values between 0 and 1, summing to 1. It's the core operation inside every attention layer in transformers (GPT, LLaMA, Claude, etc.), appearing after the Q×K^T matmul to normalize attention scores.

The math for each row:

```
softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))   for all j in row
```

The numerically stable algorithm:

```
1. max    = max(row)              ← prevent overflow in exp
2. shifted = row - max            ← all values ≤ 0
3. exp_vals = exp(shifted)        ← all values in (0, 1]
4. sum    = sum(exp_vals)
5. result = exp_vals / sum        ← normalize to sum to 1
```

**Why write it as a kernel?** In a standard framework, these are 4 separate operations, each reading from and writing back to off-chip memory (HBM). A fused NKI kernel does all steps on-chip in SBUF without round-tripping to HBM between steps — significantly faster.

In this workshop, we'll build up to softmax one concept at a time, then finish with a fused matmul+bias+ReLU to cover the Tensor Engine.

## The Mental Model

Every NKI kernel follows the same three-phase pattern:

```
HBM (off-chip)  ──dma_copy──>  SBUF (on-chip)  ──compute──>  SBUF  ──dma_copy──>  HBM (output)
```

- **HBM** — large, slow, off-chip memory where your PyTorch tensors live
- **SBUF** — small (24–32 MiB), fast, on-chip scratchpad with 128 physical partitions
- **Compute** — happens entirely on data already in SBUF (Vector Engine, Scalar Engine, Tensor Engine)

## The Three Compute Engines

A NeuronCore has three independent compute engines that can run **in parallel**. Understanding which engine executes which instruction is key to writing fast kernels.

```
+--------------------------------------------------+
|                  NeuronCore                      |
|                                                  |
|    +------------------------+                    |
|    |         SBUF           |                    |
|    |    (128 partitions)    |                    |
|    +---+--------+--------+--+                    |
|        |        |        |                       |
|        v        v        v                       |
|    +-------+ +-------+ +---------+               |
|    | VecE  | | SclE  | | TensorE |               |
|    +-------+ +-------+ +----+----+               |
|                              |                   |
|                              v                   |
|                         +---------+              |
|                         |  PSUM   |              |
|                         +---------+              |
+--------------------------------------------------+
        ^                      |
        |      DMA Engine      |
        v                      v
+--------------------------------------------------+
|                 HBM (Device Memory)              |
+--------------------------------------------------+
```

### Vector Engine (VecE)

The workhorse for elementwise and reduction operations. Operates on full SBUF tiles.

| Instruction | What it does |
|-------------|-------------|
| `nisa.tensor_scalar` | Apply a scalar op to every element: `y = x + 1.0` |
| `nisa.tensor_tensor` | Binary op between two tiles: `z = a + b` |
| `nisa.tensor_reduce` | Reduce a dimension: row max, row sum |
| `nisa.memset` | Fill a tile with a constant value |

### Scalar Engine (SclE)

Handles transcendental/non-linear functions. Computes `op(data * scale + bias)`.

| Instruction | What it does |
|-------------|-------------|
| `nisa.activation` | `exp`, `log`, `tanh`, `sigmoid`, `gelu` |

The `bias` parameter broadcasts from `(TILE, 1)` to `(TILE, hidden)`, making it perfect for fusing subtract + exp in softmax.

### Tensor Engine (TensorE)

A systolic array for matrix multiplication. Results land in **PSUM** (a dedicated accumulator buffer), not SBUF.

| Instruction | What it does |
|-------------|-------------|
| `nisa.nc_matmul` | Matrix multiply: `acc += stationary.T @ moving` |
| `nisa.tensor_copy` | Move data from PSUM back to SBUF |

Key constraints:
- Both inputs need the **K** (contraction) dimension on the partition axis
- Stationary input: max free dim = 128 (`nl.tile_size.gemm_stationary_fmax`)
- Moving input: max free dim = 512 (`nl.tile_size.gemm_moving_fmax`)
- Multiple `nc_matmul` calls to the same `dst` accumulate (for K-tiling)

### Data Movement (DMA Engine)

Transfers data between HBM and SBUF. Runs independently of compute engines — the compiler overlaps DMA with compute when possible.

| Instruction | What it does |
|-------------|-------------|
| `nisa.dma_copy` | HBM → SBUF (load) or SBUF → HBM (store) |

> **Note:** NeuronCore-v2/v3 also has a **GpSimd Engine** (8 programmable 512-bit processors for custom C++ operators), but it's not used through the NKI ISA-path API covered in this workshop.
>
> Source: [NeuronCore-v2 Architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuron-core-v2.html), [NKI Programming Model](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/programming_model.html)

## What We'll Build

| Step | Kernel | New Concept |
|------|--------|-------------|
| 1 | Add scalar | `@nki.jit`, `dma_copy`, `tensor_scalar`, `device_print` |
| 2 | Add two tensors | `tensor_tensor` |
| 3 | Tiled add (1D) | `affine_range`, partition tiling |
| 4 | 2D tiling | Free-dimension tiling (no compute) |
| 5 | Exp | `activation` (ScalarE) |
| 6 | Row-wise max | `tensor_reduce` |
| 7 | Simple softmax | Combining all ops |
| 8 | Full tiled softmax | `sequential_range`, multi-pass |
| 9 | Fused matmul+bias+ReLU | Tensor Engine, `nc_matmul`, PSUM |

Each step has:
- An explanation of the new concept
- A complete Python script you can save and run (e.g., `python step1_add_scalar.py`)
- Built-in validation that prints PASS/FAIL

---

## Environment Setup

Before writing kernels, you need a Trn2/Trn3 instance with the Neuron SDK installed. The easiest path is using the pre-built Deep Learning AMI with PyTorch Neuron.

> Full setup reference: [How to set up your environment for NKI development](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/setup-env.html)

### Launch a Trn2 Instance with the Deep Learning AMI

1. From the EC2 Console, choose **Launch Instance**
2. In the AMI search, search for `Deep Learning AMI Neuron PyTorch`
3. Select **Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)** (latest date)
4. Choose a **Trn2** or **Trn3** instance type (e.g., `trn2.3xlarge`)
5. Set your primary EBS volume to at least **512 GB**
6. Launch and SSH into the instance

Once connected, activate the PyTorch venv that already includes NKI:

```bash
# Activate the PyTorch venv (NKI is pre-installed)
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
```

### Verify Your Environment

```bash
python -c 'import nki; print("NKI OK")'
python -c 'import torch; print(f"PyTorch {torch.__version__}")'
```

Both should return without errors. If `import nki` fails, make sure you have a PyTorch or JAX venv activated (your shell prompt should show the venv name).

---

## Workshop Setup

Verify everything works by running `step0_setup.py`:

```python
"""step0_setup.py — Verify your NKI environment is ready."""
import os
os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn2'

import nki
import nki.isa as nisa
import nki.language as nl
import math
import torch

# device = torch.device('neuron')  # Use this for newer Neuron SDK versions
import torch_xla
device = torch_xla.device()
t = torch.zeros(1, dtype=torch.bfloat16).to(device)
del t

print(f"Platform: {os.environ['NEURON_PLATFORM_TARGET_OVERRIDE']}")
print(f"Torch: {torch.__version__}")
print(f"Neuron device: OK")
print(f"\nReady to write kernels!")
```

Run it:

```bash
python step0_setup.py
```

---

## Step 1: Hello NKI — Read, Add Scalar, Write Back

Our first kernel does the simplest possible thing: load a tensor from HBM into SBUF, add `1.0` to every element, and write the result back to HBM.

### Key concepts

- `@nki.jit` — decorator that compiles the function for NeuronCore when called with device tensors
- `nl.ndarray((rows, cols), dtype=..., buffer=nl.sbuf)` — allocate an on-chip tile in SBUF
- `nl.ndarray(shape, dtype=..., buffer=nl.shared_hbm)` — allocate the output tensor in HBM
- `nisa.dma_copy(dst=sbuf_tile, src=hbm_tensor[slice])` — load data from HBM to SBUF
- `nisa.dma_copy(dst=hbm_out[slice], src=sbuf_tile)` — store data from SBUF to HBM
- `nisa.tensor_scalar(dst=y, data=x, op0=nl.add, operand0=1.0)` — add a scalar to every element

### Debugging tip: `nl.device_print`

When something goes wrong, you can inspect values while they're on-chip:

```python
nl.device_print("label", tile[row, col])   # prints a single element
nl.device_print("tile_0_0", x[0, 0])       # confirm data arrived correctly
```

This prints during compilation/execution — it's the `printf` of NKI. Use it freely while developing, remove when done.

### Shape: `(128, 512)` bfloat16

This shape fits entirely in one SBUF tile (128 = max partition size), so no loops are needed. We focus purely on the structure.

### Pattern

```
HBM ──dma_copy──> SBUF(x) ──tensor_scalar(+1.0)──> SBUF(y) ──dma_copy──> HBM(out)
```

### Your kernel

Run it:

```bash
python step1_add_scalar.py
```

```python
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


# === Validation ===
if __name__ == "__main__":
    torch.manual_seed(42)
    # device = torch.device('neuron')  # Use this for newer Neuron SDK versions
    import torch_xla
    device = torch_xla.device()

    a = torch.randn((128, 512), dtype=torch.bfloat16)
    ref = a.float() + 1.0

    out = nki_add_scalar_kernel(a.to(device)).cpu().float()

    passed = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    max_diff = float((out - ref).abs().max())
    print(f"{'PASS' if passed else 'FAIL'}  max_diff={max_diff:.6f}")
```

---

## Step 2: Binary Elementwise — Add Two Tensors

Now we take **two** input tensors and add them element-wise. The structure is the same as Step 1, but we load two tiles and use `nisa.tensor_tensor` instead of `nisa.tensor_scalar`.

### Key concepts

- `nisa.tensor_tensor(dst=result, data1=a_tile, data2=b_tile, op=nl.add)` — element-wise binary operation between two SBUF tiles
- Multiple HBM inputs — kernels can take any number of tensor arguments
- Other ops available: `nl.subtract`, `nl.multiply`, `nl.maximum`, `nl.minimum`

### Shape: `(128, 512)` bfloat16

Still fits in one tile — no loops yet.

### Pattern

```
HBM(a) ──dma_copy──> SBUF(a_tile) ─┐
                                     ├─ tensor_tensor(add) ──> SBUF(result) ──dma_copy──> HBM(out)
HBM(b) ──dma_copy──> SBUF(b_tile) ─┘
```

### Your kernel

Run it:

```bash
python step2_add_tensors.py
```

```python
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


# === Validation ===
if __name__ == "__main__":
    torch.manual_seed(42)
    # device = torch.device('neuron')  # Use this for newer Neuron SDK versions
    import torch_xla
    device = torch_xla.device()

    a = torch.randn((128, 512), dtype=torch.bfloat16)
    b = torch.randn((128, 512), dtype=torch.bfloat16)
    ref = a.float() + b.float()

    out = nki_add_tensors_kernel(a.to(device), b.to(device)).cpu().float()

    passed = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    max_diff = float((out - ref).abs().max())
    print(f"{'PASS' if passed else 'FAIL'}  max_diff={max_diff:.6f}")
```

---

## Step 3: Partition Tiling — Handle Arbitrary Row Counts

SBUF has exactly **128 physical partitions**. The first dimension of every SBUF tile maps to these partitions, so it can never exceed 128. To process a tensor with more than 128 rows, we must **tile** — process it in chunks of 128.

```
HBM tensor [512, 512]
  ├─ tile 0: rows   0-127  ──> SBUF [128, 512] ──compute──> store to HBM[0:128]
  ├─ tile 1: rows 128-255  ──> SBUF [128, 512] ──compute──> store to HBM[128:256]
  ├─ tile 2: rows 256-383  ──> SBUF [128, 512] ──compute──> store to HBM[256:384]
  └─ tile 3: rows 384-511  ──> SBUF [128, 512] ──compute──> store to HBM[384:512]
```

### Key concepts

- `TILE = nl.tile_size.pmax` — hardware constant = 128 (max partition tile size)
- `nl.affine_range(n)` — loop construct telling the compiler iterations are **independent** (can be pipelined)
- `math.ceil(num_rows / TILE)` — number of tiles needed
- Boundary handling: `row_end = min(num_rows, row_start + TILE)`
- SBUF tiles are always allocated at the **full** TILE size: `nl.ndarray((TILE, hidden_dim), ...)`

### Shape: `(512, 512)` bfloat16 — 4 partition tiles

Same add-two-tensors operation as Step 2, but now with a tiling loop.

### Pattern

```
for each row tile i (affine_range):
  HBM(a[i]) ──dma_copy──> SBUF(a_tile) ─┐
                                          ├─ tensor_tensor(add) ──> SBUF(result) ──dma_copy──> HBM(out[i])
  HBM(b[i]) ──dma_copy──> SBUF(b_tile) ─┘
```

### Your kernel

Run it:

```bash
python step3_tiled_add.py
```

```python
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


# === Validation ===
if __name__ == "__main__":
    torch.manual_seed(42)
    # device = torch.device('neuron')  # Use this for newer Neuron SDK versions
    import torch_xla
    device = torch_xla.device()

    # Test 1: clean multiple of 128
    a = torch.randn((512, 512), dtype=torch.bfloat16)
    b = torch.randn((512, 512), dtype=torch.bfloat16)
    ref = a.float() + b.float()
    out = nki_add_tensors_tiled_kernel(a.to(device), b.to(device)).cpu().float()
    passed = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    max_diff = float((out - ref).abs().max())
    print(f"(512, 512): {'PASS' if passed else 'FAIL'}  max_diff={max_diff:.6f}")

    # Test 2: non-multiple of 128 (boundary handling)
    torch.manual_seed(99)
    a2 = torch.randn((300, 512), dtype=torch.bfloat16)
    b2 = torch.randn((300, 512), dtype=torch.bfloat16)
    ref2 = a2.float() + b2.float()
    out2 = nki_add_tensors_tiled_kernel(a2.to(device), b2.to(device)).cpu().float()
    passed2 = torch.allclose(out2, ref2, atol=1e-2, rtol=1e-2)
    max_diff2 = float((out2 - ref2).abs().max())
    print(f"(300, 512): {'PASS' if passed2 else 'FAIL'}  max_diff={max_diff2:.6f}")
```

---

## Step 4: 2D Tiling — Tile Both Dimensions

So far we've only tiled the partition dimension (rows). But the free dimension (columns) also has hardware limits — SBUF tiles can't be arbitrarily wide. In practice you'll often need to tile both dimensions.

This step has **no compute** — just load and store. The point is to isolate the double-loop pattern before using it in a real algorithm (Step 8).

```
HBM tensor [256, 1024]
  ├─ (row tile 0, col tile 0): [128, 512] ──> SBUF ──> HBM
  ├─ (row tile 0, col tile 1): [128, 512] ──> SBUF ──> HBM
  ├─ (row tile 1, col tile 0): [128, 512] ──> SBUF ──> HBM
  └─ (row tile 1, col tile 1): [128, 512] ──> SBUF ──> HBM
```

### Key concepts

- Two nested `nl.affine_range` loops — outer over partition tiles, inner over free tiles
- Both loops are independent (no loop-carried dependencies), so both use `affine_range`
- Tile sizes: `TILE_P = nl.tile_size.pmax` (128) for partition, `TILE_F = 512` for free
- This pattern assumes dimensions are exact multiples of tile sizes (no boundary handling)

### Shape: `(256, 1024)` bfloat16 — 2×2 = 4 tiles total

### Pattern

```
for each row tile p (affine_range):
  for each col tile f (affine_range):
    HBM(a[p,f]) ──dma_copy──> SBUF(tile) ──dma_copy──> HBM(out[p,f])
```

### Your kernel

Run it:

```bash
python step4_2d_tiling.py
```

```python
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


# === Validation ===
if __name__ == "__main__":
    torch.manual_seed(42)
    # device = torch.device('neuron')  # Use this for newer Neuron SDK versions
    import torch_xla
    device = torch_xla.device()

    a = torch.randn((256, 1024), dtype=torch.bfloat16)
    ref = a.float()

    out = nki_2d_tiling_kernel(a.to(device)).cpu().float()

    passed = torch.allclose(out, ref, atol=0, rtol=0)
    max_diff = float((out - ref).abs().max())
    print(f"{'PASS' if passed else 'FAIL'}  max_diff={max_diff:.6f}  (should be exactly 0 — no compute)")
```

---

## Step 5: Activation — Element-wise exp()

The Scalar Engine (ScalarE) handles transcendental functions like `exp`, `log`, `tanh`, and `sigmoid`. We access it through `nisa.activation`.

### Key concepts

- `nisa.activation(dst=y, op=nl.exp, data=x, bias=0.0, scale=1.0)` — computes `exp(x * scale + bias)`
- The `bias` and `scale` parameters shape the input before the activation: `op(data * scale + bias)`
- For plain `exp(x)`: use `scale=1.0, bias=0.0`
- Available ops: `nl.exp`, `nl.log`, `nl.tanh`, `nl.sigmoid`, `nl.gelu`

> **Note (trn1 only):** On NeuronCore-v2 (trn1), `bias` must be a `[N, 1]` SBUF tile, not a scalar. On trn2/trn3 (NCV3+), a scalar `0.0` works fine. Since this workshop targets trn2, we use the simpler scalar form.

### Shape: `(256, 512)` bfloat16 — 2 partition tiles

We combine the tiling pattern from Step 3 with the new `nisa.activation` instruction.

### Pattern

```
for each row tile i (affine_range):
  HBM(a[i]) ──dma_copy──> SBUF(x) ──activation(exp)──> SBUF(y) ──dma_copy──> HBM(out[i])
```

### Your kernel

Run it:

```bash
python step5_exp.py
```

```python
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


# === Validation ===
if __name__ == "__main__":
    torch.manual_seed(42)
    # device = torch.device('neuron')  # Use this for newer Neuron SDK versions
    import torch_xla
    device = torch_xla.device()

    a = torch.randn((256, 512), dtype=torch.bfloat16)
    a = a.clamp(-5.0, 5.0)  # prevent overflow in exp
    ref = a.float().exp()

    out = nki_exp_kernel(a.to(device)).cpu().float()

    passed = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    max_diff = float((out - ref).abs().max())
    print(f"{'PASS' if passed else 'FAIL'}  max_diff={max_diff:.6f}")
```

---

## Step 6: Row-wise Reduction — Max of Each Row

Reductions collapse an entire dimension into a single value per row. This is the foundation of softmax's numerical stability trick: we need to find the maximum value in each row before exponentiating.

### Key concepts

- `nisa.tensor_reduce(dst=result, data=tile, op=nl.maximum, axis=(1,))` — reduce along the free (hidden) dimension
- Output shape: if input is `(TILE, hidden_dim)`, output is `(TILE, 1)` — one value per partition/row
- `axis=(1,)` — always a tuple. Reduces along dimension 1 (free/hidden). Axis 0 (partition) cannot be reduced.
- Other reduction ops: `nl.add` (sum), `nl.minimum`
- The `(TILE, 1)` output will later broadcast naturally when combined with `(TILE, hidden_dim)` tiles via `nisa.tensor_tensor`

### Shape: `(256, 512)` bfloat16 → output `(256, 1)` float32

### Pattern

```
for each row tile i (affine_range):
  HBM(a[i]) ──dma_copy──> SBUF(x) ──tensor_reduce(max, axis=1)──> SBUF(row_max) ──dma_copy──> HBM(out[i])
```

### Your kernel

Run it:

```bash
python step6_row_max.py
```

```python
"""step6_row_max.py — Row-wise max reduction."""
import os
os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn2'

import math
import nki
import nki.isa as nisa
import nki.language as nl
import torch

@nki.jit
def nki_row_max_kernel(a):
    num_rows, hidden_dim = a.shape
    TILE = nl.tile_size.pmax
    num_tiles = math.ceil(num_rows / TILE)

    out = nl.ndarray((num_rows, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    for i in nl.affine_range(num_tiles):
        row_start = i * TILE
        row_end = min(num_rows, row_start + TILE)
        tile_h = row_end - row_start

        x = nl.ndarray((TILE, hidden_dim), dtype=a.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x[0:tile_h, 0:hidden_dim], src=a[row_start:row_end, 0:hidden_dim])

        row_max = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=row_max, data=x, op=nl.maximum, axis=(1,))

        nisa.dma_copy(dst=out[row_start:row_end, 0:1], src=row_max[0:tile_h, 0:1])

    return out


# === Validation ===
if __name__ == "__main__":
    torch.manual_seed(42)
    # device = torch.device('neuron')  # Use this for newer Neuron SDK versions
    import torch_xla
    device = torch_xla.device()

    a = torch.randn((256, 512), dtype=torch.bfloat16)
    ref = a.float().max(dim=-1, keepdim=True).values  # (256, 1)

    out = nki_row_max_kernel(a.to(device)).cpu().float()

    passed = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    max_diff = float((out - ref).abs().max())
    print(f"{'PASS' if passed else 'FAIL'}  max_diff={max_diff:.6f}")
```

---

## Step 7: Simple Softmax — All Ops Combined

Now we combine everything from Steps 1–6 into a complete softmax kernel. The algorithm:

```
For each row:
  1. row_max = max(row)                    ← numerical stability
  2. shifted = row - row_max               ← all values ≤ 0
  3. exp_vals = exp(shifted)               ← all values in (0, 1]
  4. row_sum = sum(exp_vals)
  5. result = exp_vals / row_sum           ← normalize to sum to 1
```

### Key concepts

- Combining multiple ISA instructions into one algorithm
- Broadcasting: `(TILE, 1)` operands broadcast against `(TILE, hidden_dim)` tiles automatically
- `nisa.activation` can fuse subtract + exp into one call via its `bias` parameter: `exp(x + bias)` where `bias = -row_max`
- `exp(x - max - log(sum))` = `exp(x - max) / sum` — the log-sum-exp trick avoids a separate division

### Shape: `(128, 512)` bfloat16 — one partition tile, hidden fits in one tile

### Pattern

```
HBM(a) ──dma_copy──> SBUF(x)
  ├── tensor_reduce(max) ──> SBUF(row_max)
  ├── negate(row_max) ──> SBUF(neg_row_max)
  ├── activation(exp, bias=neg_row_max) ──> SBUF(exp_vals)
  ├── tensor_reduce(sum) ──> SBUF(row_sum)
  ├── activation(log) ──> SBUF(log_sum)
  ├── tensor_tensor(subtract) ──> SBUF(combined_bias)
  └── activation(exp, bias=combined_bias) ──> SBUF(result) ──dma_copy──> HBM(out)
```

### Your kernel

Run it:

```bash
python step7_softmax_simple.py
```

```python
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


# === Validation ===
if __name__ == "__main__":
    torch.manual_seed(42)
    # device = torch.device('neuron')  # Use this for newer Neuron SDK versions
    import torch_xla
    device = torch_xla.device()

    a = torch.randn((128, 512), dtype=torch.bfloat16)
    ref = torch.softmax(a.float(), dim=-1)

    out = nki_softmax_simple_kernel(a.to(device)).cpu().float()

    passed = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    max_diff = float((out - ref).abs().max())
    print(f"{'PASS' if passed else 'FAIL'}  max_diff={max_diff:.6f}")

    # Sanity check: rows should sum to 1.0
    row_sums = out.sum(dim=-1)
    print(f"Row sums — min: {row_sums.min():.4f}  max: {row_sums.max():.4f}  (should be ~1.0)")
```

---

## Step 8: Full Tiled Softmax — The Final Boss

In real models, the hidden dimension can be much larger than one tile (e.g., 4096 or 8192). We need to tile across the hidden dimension too. But softmax has a problem: we need the **global** row max before we can compute exp, but we can only load one hidden tile at a time.

### The solution: three sequential passes

```
For each row tile (independent — nl.affine_range):

  Pass 1 — Find row max (nl.sequential_range over hidden tiles):
    row_max = -inf
    for each hidden tile:
      local_max = reduce(tile, op=max)
      row_max = max(row_max, local_max)     ← loop-carried dependency!

  Pass 2 — Compute exp and sum (nl.sequential_range over hidden tiles):
    row_sum = 0
    for each hidden tile:
      shifted = tile - row_max
      exp_vals = exp(shifted)
      local_sum = reduce(exp_vals, op=add)
      row_sum = row_sum + local_sum          ← loop-carried dependency!

  Pass 3 — Normalize and store (nl.affine_range — independent!):
    for each hidden tile:
      result = exp(tile - row_max - log(row_sum))
      store result
```

### Key concepts

- `nl.sequential_range(n)` — loop where iteration `i` depends on `i-1`. The compiler executes these strictly in order. **Required** for passes 1 and 2.
- `nl.affine_range(n)` — iterations are independent and can be pipelined. Used for the outer row loop and pass 3.
- `nisa.memset(dst=row_max, value=-65504.0)` — initialize row_max to negative "infinity" (min finite bfloat16 value)
- `nisa.memset(dst=row_sum, value=0.0)` — initialize sum accumulator to zero
- Using `nl.affine_range` where `nl.sequential_range` is needed would give **wrong results** (the compiler would try to parallelize dependent iterations)

### Shape: `(256, 512)` bfloat16 with `HIDDEN_TILE = 128`

### Pattern

```
for each row tile (affine_range):

  Pass 1 (sequential_range):  HBM ──> SBUF ──reduce(max)──> update row_max
  Pass 2 (sequential_range):  HBM ──> SBUF ──exp(x-max)──> reduce(sum) ──> update row_sum
  Pass 3 (affine_range):      HBM ──> SBUF ──exp(x-max-log(sum))──> SBUF ──dma_copy──> HBM
```

### Your kernel

Run it:

```bash
python step8_softmax_tiled.py
```

```python
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


# === Validation ===
if __name__ == "__main__":
    torch.manual_seed(42)
    # device = torch.device('neuron')  # Use this for newer Neuron SDK versions
    import torch_xla
    device = torch_xla.device()

    # Test 1: standard shape
    a = torch.randn((256, 512), dtype=torch.bfloat16)
    ref = torch.softmax(a.float(), dim=-1)
    out = nki_softmax_kernel(a.to(device)).cpu().float()
    passed = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    max_diff = float((out - ref).abs().max())
    print(f"(256, 512): {'PASS' if passed else 'FAIL'}  max_diff={max_diff:.6f}")
    row_sums = out.sum(dim=-1)
    print(f"  Row sums min={row_sums.min():.4f} max={row_sums.max():.4f}")

    # Test 2: larger hidden dim (forces multiple hidden tiles)
    torch.manual_seed(7)
    a2 = torch.randn((128, 1024), dtype=torch.bfloat16)
    ref2 = torch.softmax(a2.float(), dim=-1)
    out2 = nki_softmax_kernel(a2.to(device)).cpu().float()
    passed2 = torch.allclose(out2, ref2, atol=1e-2, rtol=1e-2)
    max_diff2 = float((out2 - ref2).abs().max())
    print(f"(128, 1024): {'PASS' if passed2 else 'FAIL'}  max_diff={max_diff2:.6f}")
```

---

## Step 9: Fused Matmul + Bias + ReLU — The Tensor Engine

So far we've used the **Vector Engine** (elementwise ops) and **Scalar Engine** (transcendentals). The third major compute unit is the **Tensor Engine** — a systolic array that does matrix multiplication.

### Why fusion matters

Without NKI, computing `relu(A @ B + bias)` requires three separate kernel launches (matmul, add, relu), each writing results to HBM and reading them back. A fused kernel keeps data on-chip between steps:

```
HBM ──dma──> SBUF ──nc_matmul──> PSUM ──tensor_copy──> SBUF ──bias──> SBUF ──relu──> SBUF ──dma──> HBM
```

### Key concepts

- `nisa.nc_matmul(dst=acc, stationary=a_tile, moving=b_tile)` — matrix multiply on Tensor Engine
- **PSUM** — a dedicated accumulator buffer wired to the Tensor Engine output. Results land here after matmul.
- `nl.psum` — buffer type for PSUM tiles: `nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)`
- `nisa.tensor_copy(dst=sbuf_tile, src=psum_tile)` — move data from PSUM to SBUF (required — `dma_copy` can't read PSUM)
- **Stationary input** — first operand, loaded once per output tile. Shape: `(K, M)` with `K` on partition axis. Max free dim: `TILE_M = nl.tile_size.gemm_stationary_fmax` = 128.
- **Moving input** — second operand, streamed through. Shape: `(K, N)` with `K` on partition axis. Max free dim: `TILE_N = nl.tile_size.gemm_moving_fmax` = 512.
- Both inputs need K on the **partition** axis — that's why A is passed pre-transposed as AT `[K, M]`
- Multiple `nc_matmul` calls to the same `dst` accumulate (useful for tiling over K)

### Shape: AT `[256, 256]`, B `[256, 512]`, bias `[1, 512]` — all float16

The matmul part (Steps 1–3) is done for you. **Your job: complete Steps 4–5** (add bias, apply ReLU, store).

### Pattern

```
for each output tile [m, n] (affine_range):
  for each K tile (affine_range):
    HBM(AT[k,m]) ──dma_copy──> SBUF(a_tile) ─┐
                                               ├─ nc_matmul ──> PSUM(acc) [accumulates]
    HBM(B[k,n])  ──dma_copy──> SBUF(b_tile) ─┘

  PSUM(acc) ──tensor_copy──> SBUF(matmul_sbuf)
  HBM(bias) ──dma_copy──> SBUF(bias_tile) [broadcast to all partitions]
  SBUF ──tensor_tensor(add)──> SBUF(biased)
  SBUF ──tensor_scalar(max, 0)──> SBUF(relu_result) ──dma_copy──> HBM(C[m,n])
```

### Your kernel

Run it:

```bash
python step9_fused_matmul.py
```

```python
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

    # Step 1: Allocate output in HBM
    C = nl.ndarray((M, N), dtype=AT.dtype, buffer=nl.shared_hbm)

    # Step 2: Outer loops tile the output; inner loop accumulates over K
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

            # Step 3: Move result from PSUM to SBUF
            matmul_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=AT.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=matmul_sbuf, src=acc)

            # Step 4: Add bias
            # bias is (1, TILE_N) but matmul_sbuf is (TILE_M, TILE_N).
            # tensor_tensor requires matching partition dimensions.
            # Solution: copy bias into all TILE_M partitions to make (TILE_M, TILE_N).
            bias_tile = nl.ndarray((TILE_M, TILE_N), dtype=bias.dtype, buffer=nl.sbuf)
            for i_p in nl.affine_range(TILE_M):
                nisa.dma_copy(dst=bias_tile[i_p:i_p+1, 0:TILE_N],
                              src=bias[0:1, i_n*TILE_N:(i_n+1)*TILE_N])

            biased = nl.ndarray((TILE_M, TILE_N), dtype=AT.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=biased, data1=matmul_sbuf, data2=bias_tile, op=nl.add)

            # Step 5: Apply ReLU and store
            relu_result = nl.ndarray((TILE_M, TILE_N), dtype=AT.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=relu_result, data=biased, op0=nl.maximum, operand0=0.0)

            nisa.dma_copy(dst=C[i_m*TILE_M:(i_m+1)*TILE_M, i_n*TILE_N:(i_n+1)*TILE_N],
                          src=relu_result)

    return C


# === Validation ===
if __name__ == "__main__":
    torch.manual_seed(0)
    # device = torch.device('neuron')  # Use this for newer Neuron SDK versions
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
    print(f"{'PASS' if passed else 'FAIL'}  max_diff={max_diff:.6f}")
    print(f"  Output shape: {out.shape}  min={out.min():.4f} (should be >= 0)")
```

---

## Workshop Complete

You've covered all three compute engines on Trainium:

| Engine | Instructions Used | What It Does |
|--------|------------------|--------------|
| **Vector Engine** | `tensor_scalar`, `tensor_tensor`, `tensor_reduce`, `memset` | Elementwise ops, reductions |
| **Scalar Engine** | `activation` | Transcendentals (exp, log, tanh, gelu) |
| **Tensor Engine** | `nc_matmul` | Matrix multiplication |

Plus the data movement primitives:

| Instruction | Purpose |
|-------------|---------|
| `nisa.dma_copy` | HBM ↔ SBUF transfers |
| `nisa.tensor_copy` | PSUM → SBUF (after matmul) |

And the control flow:

| Construct | When to Use |
|-----------|-------------|
| `nl.affine_range` | Independent iterations (pipelineable) |
| `nl.sequential_range` | Dependent iterations (loop-carried state) |

### What's Next

- **Profile your kernels** — use `neuron-profile` to see how the engines overlap
- **Try larger shapes** — real models use hidden dims of 4096–8192
- **Write your own** — Layer Norm, RMS Norm, or fused attention are natural next steps
- **Explore** — see the `kernels/` directory for more reference implementations
