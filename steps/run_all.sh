#!/bin/bash
# Run all tutorial steps sequentially on a Trainium instance.
# Usage: bash run_all.sh

set -e

echo "=========================================="
echo "NKI Tutorial — Running all steps"
echo "=========================================="

for step in step0_setup.py step1_add_scalar.py step2_add_tensors.py step3_tiled_add.py \
            step4_2d_tiling.py step5_exp.py step6_row_max.py step7_softmax_simple.py \
            step8_softmax_tiled.py step9_fused_matmul.py; do
    echo ""
    echo "--- Running $step ---"
    python "$step"
done

echo ""
echo "=========================================="
echo "All steps complete!"
echo "=========================================="
