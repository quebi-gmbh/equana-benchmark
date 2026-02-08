/**
 * Single translation unit build - includes all kernel code directly
 * This allows full inlining of micro-kernels into the driver.
 */

/* Include level3 driver */
#include "level3.c"

/* Include all kernels - they become part of this TU */
#include "kernel/dgemm_kernel_4x4_wasm.c"
#include "kernel/dgemm_kernel_sse_4x4_wasm.c"
#include "kernel/dgemm_kernel_native_4x4_wasm.c"
