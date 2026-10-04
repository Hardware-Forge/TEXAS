#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>


#define THREADS 32
#define TEX_SIZE (1 << 18)     // 256K
#define BLOCK_SIZE 100000
#define TOTAL_ITERS 5000000
#define LABEL_ID -1  // 0=CNN 1=3D 2=transformer 3=Diffusion 4=RNN  5=Mining use only if need to label measurements (-1 for no label)
#define OUTPUT_CSV "tex_spy_timings.csv" // Output CSV file path



// ============================================================================
//  SPY Kernel - Streaming with offset
// ============================================================================
__global__ void spyTexKernelStreaming(unsigned long long* timings,
                                      cudaTextureObject_t spyTexObj,
                                      int start_iter, int block_size)
{
    int tid = threadIdx.x;
    __shared__ unsigned long long sh_times[32];
    __shared__ volatile int local_vals[32];

    // Initialize LFSR with offset
    unsigned int lfsr = tid * 7919 + 104729 + start_iter;

    for (int i = 0; i < block_size; i++) {
        lfsr = (lfsr >> 1) ^ (-(lfsr & 1) & 0xD0000001u);
        int idx = lfsr & (TEX_SIZE - 1);

        __syncthreads();

        // Timing measurement
        unsigned long long start = clock64();
        int val = tex1Dfetch<int>(spyTexObj, idx);
        local_vals[tid] = val;
        unsigned long long stop = clock64();

        sh_times[tid] = stop - start;

        __syncthreads();

        if (tid == 0) {
            unsigned long long temp[32];
            for (int t = 0; t < 32; t++) temp[t] = sh_times[t];

            // Full sort for median
            for (int a = 0; a < 32; a++) {
                for (int b = a + 1; b < 32; b++) {
                    if (temp[a] > temp[b]) {
                        unsigned long long swap = temp[a];
                        temp[a] = temp[b];
                        temp[b] = swap;
                    }
                }
            }
            timings[i] = temp[16]; // median
        }

        if ((i & 0x7) == 0x7) __syncthreads();
    }
}

// ============================================================================
//  Utility functions
// ============================================================================
cudaTextureObject_t createTexObject(int* d_ptr)
{
    cudaResourceDesc res = {};
    res.resType = cudaResourceTypeLinear;
    res.res.linear.devPtr = d_ptr;
    res.res.linear.sizeInBytes = TEX_SIZE * sizeof(int);
    res.res.linear.desc = cudaCreateChannelDesc<int>();

    cudaTextureDesc tex = {};
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModePoint;
    tex.readMode = cudaReadModeElementType;
    tex.normalizedCoords = 0;

    cudaTextureObject_t obj;
    cudaCreateTextureObject(&obj, &res, &tex, nullptr);
    return obj;
}

void sortArray(unsigned long long* arr, int n)
{
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (arr[i] > arr[j]) {
                unsigned long long temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
}

double calculateMedian(unsigned long long* data, int n)
{
    unsigned long long* sorted = new unsigned long long[n];
    for (int i = 0; i < n; i++) sorted[i] = data[i];
    sortArray(sorted, n);
    double median = sorted[n/2];
    delete[] sorted;
    return median;
}

void printStats(const char* label, unsigned long long* data, int n)
{
    unsigned long long* sorted = new unsigned long long[n];
    for (int i = 0; i < n; i++) sorted[i] = data[i];
    sortArray(sorted, n);

    unsigned long long median = sorted[n/2];
    unsigned long long p25 = sorted[n/4];
    unsigned long long p75 = sorted[3*n/4];
    unsigned long long p95 = sorted[95*n/100];

    // Trimmed mean (exclude top/bottom 5%)
    unsigned long long sum = 0;
    int start = 5*n/100;
    int end = 95*n/100;
    for (int i = start; i < end; i++) sum += sorted[i];
    double avg = double(sum) / (end - start);

    printf("%s:\n", label);
    printf("  Median: %5llu | Trimmed avg: %6.1f | IQR: %4llu | P95: %5llu\n",
           median, avg, p75 - p25, p95);

    delete[] sorted;
}


// ============================================================================
// MAIN - Streaming
// ============================================================================
int main()
{

    // Setup GPU
    int *d_spy_data;
    cudaMalloc(&d_spy_data, TEX_SIZE * sizeof(int));
    int* h_data = new int[TEX_SIZE];
    for (int i = 0; i < TEX_SIZE; i++)
        h_data[i] = i * 7919 + 104729;
    cudaMemcpy(d_spy_data, h_data, TEX_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    delete[] h_data;

    cudaTextureObject_t spyTex = createTexObject(d_spy_data);

    // Device buffer per block

    unsigned long long *d_block;
    cudaMalloc(&d_block, BLOCK_SIZE * sizeof(unsigned long long));
    unsigned long long* h_block = new unsigned long long[BLOCK_SIZE];

    // CSV
    FILE* f = fopen(OUTPUT_CSV, "w");
    if (!f) {
        printf("Error opening CSV file!\n");
        return -1;
    }
    fprintf(f, "iteration,latency_cycles,class\n");


    int num_blocks = (TOTAL_ITERS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("Executing streaming run: total iterations %d, block size %d (%d blocks)\n",
           TOTAL_ITERS, BLOCK_SIZE, num_blocks);

    int iter_count = 0;
    for (int blk = 0; blk < num_blocks; blk++) {
        int this_block_size = BLOCK_SIZE;
        if (iter_count + this_block_size > TOTAL_ITERS)
            this_block_size = TOTAL_ITERS - iter_count;

        // Launch kernel with offset
        spyTexKernelStreaming<<<1, THREADS>>>(d_block, spyTex, iter_count, this_block_size);
        cudaDeviceSynchronize();

        // Copy results to host
        cudaMemcpy(h_block, d_block, this_block_size * sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);

        // Write to CSV
        if (LABEL_ID < 0) {
            for (int i = 0; i < this_block_size; i++) {
                fprintf(f, "%d,%llu\n", iter_count + i, h_block[i]);
            }
        } else {
            for (int i = 0; i < this_block_size; i++) {
                fprintf(f, "%d,%llu,%d\n", iter_count + i, h_block[i], LABEL_ID);
            }
        }

        iter_count += this_block_size;
        printf("\rProgress: %.2f%%", 100.0 * iter_count / TOTAL_ITERS);
        fflush(stdout);
    }
    printf("\n");

    fclose(f);
    printf("\nResults saved to " OUTPUT_CSV "\n");

    // Cleanup
    delete[] h_block;
    cudaFree(d_block);
    cudaDestroyTextureObject(spyTex);
    cudaFree(d_spy_data);

    printf("\nTest completed!\n");
    return 0;
}