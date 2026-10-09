import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    out_ptr, x_ptr,
    x_row_stride, out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    row_start_ptr = x_ptr + row_idx * x_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    row_max = tl.max(row, axis=0)
    row_minus_max = row - row_max

    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator

    out_row_start_ptr = out_ptr + row_idx * out_row_stride
    output_ptrs = out_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_out, mask=mask)

def softmax_triton(x, num_warps=4, return_compiled=False):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)

    compiled = softmax_kernel[(n_rows,)](
        out, x,
        x.stride(0), out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    if return_compiled:
        return out, compiled
    return out