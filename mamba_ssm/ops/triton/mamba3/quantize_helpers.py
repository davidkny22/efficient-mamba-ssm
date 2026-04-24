"""Block-scaled quantization helpers for tl.dot_scaled."""

import triton
import triton.language as tl


@triton.jit
def quantize_block_e4m3(x, BLOCK_SCALE_SIZE: tl.constexpr = 16):
    M = x.shape[0]
    N = x.shape[1]
    NUMEL = M * N
    NUM_BLOCKS = NUMEL // BLOCK_SCALE_SIZE

    x_blocks = tl.reshape(x.to(tl.float32), [NUM_BLOCKS, BLOCK_SCALE_SIZE])
    absmax = tl.max(tl.abs(x_blocks), axis=1)
    scales = tl.maximum(absmax / 448.0, 1.0e-12)

    x_scaled = x_blocks / scales[:, None]
    x_scaled = tl.minimum(tl.maximum(x_scaled, -448.0), 448.0)
    x_quantized = tl.reshape(x_scaled.to(tl.float8e4m3fn), [M, N])

    return x_quantized, scales.to(tl.float8e4m3fn)


@triton.jit
def quantize_block_e2m1(x, BLOCK_SCALE_SIZE: tl.constexpr = 16):
    M = x.shape[0]
    N = x.shape[1]
    NUMEL = M * N
    NUM_BLOCKS = NUMEL // BLOCK_SCALE_SIZE

    x_blocks = tl.reshape(x.to(tl.float32), [NUM_BLOCKS, BLOCK_SCALE_SIZE])
    absmax = tl.max(tl.abs(x_blocks), axis=1)
    scales = tl.maximum(absmax / 6.0, 1.0e-12)

    x_scaled = x_blocks / scales[:, None]
    x_scaled = tl.minimum(tl.maximum(x_scaled, -6.0), 6.0)
    abs_x = tl.abs(x_scaled)

    fp4_code = tl.where(
        abs_x < 0.25,
        0,
        tl.where(
            abs_x < 0.75,
            1,
            tl.where(
                abs_x < 1.25,
                2,
                tl.where(
                    abs_x < 1.75,
                    3,
                    tl.where(
                        abs_x < 2.5,
                        4,
                        tl.where(abs_x < 3.5, 5, tl.where(abs_x < 5.0, 6, 7)),
                    ),
                ),
            ),
        ),
    ).to(tl.uint8)
    sign_bit = tl.where(x_scaled < 0.0, 8, 0).to(tl.uint8)
    fp4_code = fp4_code | sign_bit

    fp4_code = tl.reshape(fp4_code, [M, N // 2, 2])
    # NOTE: verify tl.split splits along last axis (dim=2) — if not, packing is wrong
    lo, hi = tl.split(fp4_code)
    x_packed = lo | (hi << 4)

    return x_packed, scales.to(tl.float8e4m3fn)


@triton.jit
def quantize_block_e5m2(x, BLOCK_SCALE_SIZE: tl.constexpr = 16):
    M = x.shape[0]
    N = x.shape[1]
    NUMEL = M * N
    NUM_BLOCKS = NUMEL // BLOCK_SCALE_SIZE

    x_blocks = tl.reshape(x.to(tl.float32), [NUM_BLOCKS, BLOCK_SCALE_SIZE])
    absmax = tl.max(tl.abs(x_blocks), axis=1)
    scales = tl.maximum(absmax / 57344.0, 1.0e-12)

    x_scaled = x_blocks / scales[:, None]
    x_scaled = tl.minimum(tl.maximum(x_scaled, -57344.0), 57344.0)
    x_quantized = tl.reshape(x_scaled.to(tl.float8e5m2), [M, N])

    return x_quantized, scales.to(tl.float8e4m3fn)
