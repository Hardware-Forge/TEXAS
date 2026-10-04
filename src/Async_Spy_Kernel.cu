#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS 32
#define DATA_SIZE_BYTES (1LL << 28) // 256 MB
#define DATA_SIZE_INTS (DATA_SIZE_BYTES / sizeof(int))

#define BLOCK_SIZE 100000
#define TOTAL_ITERS 5000000

#define LABEL_ID -1   // 0=CNN 1=3D 2=transformer 3=Diffusion 4=RNN 5=Mining use only if need to label measurements (-1 for no label)
#define OUTPUT_CSV "async_spy_timings.csv" // Output CSV file path

// ============================================================================
//  SPY Kernel - cp.async streaming
// ============================================================================
__global__ void spyAsyncKernelStreaming(unsigned long long* timings,
                                       int* gmem,
                                       int start_iter,
                                       int block_size)
{
    __shared__ __align__(16) int smem[THREADS * 16]; // spazio sufficiente
    __shared__ unsigned long long sh_times[32];
    __shared__ volatile int local_vals[32];

    int tid = threadIdx.x;
    int base = tid * 16; // 16 int = 64 byte

    unsigned int lfsr = tid * 7919 + 104729 + start_iter;

    for (int i = 0; i < block_size; i++) {

        // Random access
        lfsr = (lfsr >> 1) ^ (-(lfsr & 1) & 0xD0000001u);
        int gmem_idx = (lfsr % (DATA_SIZE_INTS / 16)) * 16;

        unsigned long long src =
            __cvta_generic_to_global(&gmem[gmem_idx]);
        unsigned long long dst =
            __cvta_generic_to_shared(&smem[base]);

        __syncwarp();

        // ---- cp.async #1 ----
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            "cp.async.commit_group;\n"
            :: "l"(dst +  0), "l"(src +  0)
        );

        // ---- cp.async #2 ----
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            "cp.async.commit_group;\n"
            :: "l"(dst + 16), "l"(src + 16)
        );

        // ---- cp.async #3 ----
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            "cp.async.commit_group;\n"
            :: "l"(dst + 32), "l"(src + 32)
        );

        // ---- cp.async #4 ----
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            "cp.async.commit_group;\n"
            :: "l"(dst + 48), "l"(src + 48)
        );

        asm volatile("" ::: "memory");
        unsigned long long start = clock64();

        // ---- WAIT ----
        asm volatile("cp.async.wait_group 0;\n");

        asm volatile("" ::: "memory");
        unsigned long long stop = clock64();

        __syncwarp();
        volatile int x =
            smem[base +  0] ^
            smem[base +  4] ^
            smem[base +  8] ^
            smem[base + 12];

        local_vals[tid] = x;
        sh_times[tid] = stop - start;

        __syncthreads();

        // ---- median warp ----
        if (tid == 0) {
            unsigned long long tmp[32];
            for (int t = 0; t < 32; t++) tmp[t] = sh_times[t];

            for (int a = 0; a < 32; a++)
                for (int b = a + 1; b < 32; b++)
                    if (tmp[a] > tmp[b]) {
                        auto s = tmp[a];
                        tmp[a] = tmp[b];
                        tmp[b] = s;
                    }

            timings[i] = tmp[16];
        }

        __syncthreads();
    }
}

// ============================================================================
// MAIN - Streaming
// ============================================================================
int main()
{

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (prop.major < 8) {
        printf("ERROR: cp.async requires SM 8.0+\n");
        return 1;
    }

    // Allocate spy data
    int* d_data;
    cudaMalloc(&d_data, DATA_SIZE_BYTES);

    // Allocate per-block timing buffer
    unsigned long long* d_block;
    cudaMalloc(&d_block, BLOCK_SIZE * sizeof(unsigned long long));
    unsigned long long* h_block =
        new unsigned long long[BLOCK_SIZE];

    // CSV
    FILE* f = fopen(OUTPUT_CSV, "w");
    if (!f) {
        printf("Error opening CSV\n");
        return -1;
    }

    if (LABEL_ID >= 0)
        fprintf(f, "iteration,latency_cycles,class\n");
    else
        fprintf(f, "iteration,latency_cycles\n");

    int num_blocks =
        (TOTAL_ITERS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int iter_count = 0;

    printf("Streaming %d iterations (%d blocks)\n",
           TOTAL_ITERS, num_blocks);

    for (int blk = 0; blk < num_blocks; blk++) {

        int this_block = BLOCK_SIZE;
        if (iter_count + this_block > TOTAL_ITERS)
            this_block = TOTAL_ITERS - iter_count;

        spyAsyncKernelStreaming<<<1, THREADS>>>(
            d_block, d_data, iter_count, this_block);

        cudaDeviceSynchronize();

        cudaMemcpy(h_block, d_block,
                   this_block * sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);

        for (int i = 0; i < this_block; i++) {
            if (LABEL_ID >= 0)
                fprintf(f, "%d,%llu,%d\n",
                        iter_count + i, h_block[i], LABEL_ID);
            else
                fprintf(f, "%d,%llu\n",
                        iter_count + i, h_block[i]);
        }

        iter_count += this_block;
        printf("\rProgress: %.2f%%",
               100.0 * iter_count / TOTAL_ITERS);
        fflush(stdout);
    }

    printf("\nDone.\n");
    fclose(f);

    delete[] h_block;
    cudaFree(d_block);
    cudaFree(d_data);

    printf("Results saved to %s\n", OUTPUT_CSV);
    return 0;
}