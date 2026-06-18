%%writefile unify_kernel.cu
#include <cuda_runtime.h>
#include <cstdint>

__global__ void update_ref_kernel(const int32_t* subs, int32_t* out_roots, int32_t total_elements) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Minimal copy operations to test the physical VRAM pipeline
    out_roots[idx] = subs[idx];
}

void launch_update_ref_kernel(const int32_t* subs_ptr, int32_t* out_roots_ptr, int32_t num_batches, int32_t num_nodes) {
    int32_t total_elements = num_batches * num_nodes;
    
    int32_t threads = 256;
    int32_t blocks = (total_elements + threads - 1) / threads;

    update_ref_kernel<<<blocks, threads>>>(subs_ptr, out_roots_ptr, total_elements);
}