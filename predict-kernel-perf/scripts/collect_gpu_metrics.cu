#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Simple CUDA error check macro
#define CHECK_CUDA(call) do {                                 \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error %s at %s:%d\n",           \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)

// Map compute capability -> FP32 cores per SM (approx, good enough)
int get_fp32_cores_per_sm(int major, int minor) {
    int sm = major * 10 + minor;
    switch (sm) {
        case 50:  // Maxwell
        case 52:
        case 53: return 128;
        case 60: return 64;   // GP100
        case 61: return 128;  // GP102/104 etc.
        case 62: return 128;
        case 70: return 64;   // Volta
        case 72: return 64;
        case 75: return 64;   // Turing
        case 80: return 64;   // A100
        case 86: return 128;  // RTX 30-series
        case 89: return 128;  // Ada (e.g. RTX 40-series incl 4070)
        case 90: return 128;  // Hopper-ish assumption
        default: return -1;   // Unknown
    }
}

// Very rough architecture name from compute capability)
// Verified generated architecture names online after finding cuda2, cuda3, cuda4, cuda5 CIMS GPUs
const char* get_arch_name(int major, int minor) {
    int sm = major * 10 + minor;
    switch (sm) {
        case 50: case 52: case 53: return "Maxwell";
        case 60: case 61: case 62: return "Pascal";
        case 70: case 72:          return "Volta";
        case 75:                   return "Turing";
        case 80: case 86:          return "Ampere";
        case 89:                   return "Ada";
        case 90:                   return "Hopper";
        default:                   return "Unknown";
    }
}

// ---- Sustained bandwidth benchmark: device-to-device memcpy ----
double benchmark_sustained_bandwidth(size_t size_bytes = (1u << 28), int repeats = 5) {
    void* src;
    void* dst;
    CHECK_CUDA(cudaMalloc(&src, size_bytes));
    CHECK_CUDA(cudaMalloc(&dst, size_bytes));

    // Warmup
    CHECK_CUDA(cudaMemcpy(dst, src, size_bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    float best_ms = 1e30f;

    for (int i = 0; i < repeats; ++i) {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(dst, src, size_bytes, cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        if (ms < best_ms) best_ms = ms;

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    CHECK_CUDA(cudaFree(src));
    CHECK_CUDA(cudaFree(dst));

    double seconds = best_ms / 1000.0;
    double bytes_per_s = static_cast<double>(size_bytes) / seconds;
    double gb_per_s = bytes_per_s / 1e9;
    return gb_per_s;
}

// ---- Sustained compute benchmark: dense FMA kernel ----
__global__ void fma_kernel(float* a, const float* b, const float* c, int N, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float va = a[i];
        float vb = b[i];
        float vc = c[i];
        for (int k = 0; k < iters; ++k) {
            va = fmaf(va, vb, vc); // 1 FMA = 2 FLOPs
        }
        a[i] = va; // store result so compiler can't drop it
    }
}

double benchmark_sustained_compute(int N = (1 << 20), int iters = 1024, int block_size = 256) {
    size_t bytes = static_cast<size_t>(N) * sizeof(float);

    float* a;
    float* b;
    float* c;
    CHECK_CUDA(cudaMalloc(&a, bytes));
    CHECK_CUDA(cudaMalloc(&b, bytes));
    CHECK_CUDA(cudaMalloc(&c, bytes));

    // We don't care about initial values; just need valid memory.
    CHECK_CUDA(cudaMemset(a, 0, bytes));
    CHECK_CUDA(cudaMemset(b, 0, bytes));
    CHECK_CUDA(cudaMemset(c, 0, bytes));

    int grid_size = (N + block_size - 1) / block_size;

    // Warmup
    fma_kernel<<<grid_size, block_size>>>(a, b, c, N, iters);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    fma_kernel<<<grid_size, block_size>>>(a, b, c, N, iters);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(c));

    double seconds = ms / 1000.0;
    // FLOPs = N * iters * 2 (FMA)
    double flops = static_cast<double>(N) * static_cast<double>(iters) * 2.0;
    double gflops = flops / 1e9 / seconds;
    return gflops;
}

int main() {
    int dev = 0;
    CHECK_CUDA(cudaSetDevice(dev));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    const char* device_name = prop.name;
    int major = prop.major;
    int minor = prop.minor;
    const char* arch_name = get_arch_name(major, minor);

    int num_sms              = prop.multiProcessorCount;
    int warp_size            = prop.warpSize;
    int max_threads_per_sm   = prop.maxThreadsPerMultiProcessor;
    int regs_per_sm          = prop.regsPerMultiprocessor;
    int shared_mem_per_sm    = prop.sharedMemPerMultiprocessor;
    int l2_cache_size        = prop.l2CacheSize;               // may be 0 on older toolkits
    int max_blocks_per_sm    = prop.maxBlocksPerMultiProcessor;

    int sm_clock_khz         = prop.clockRate;                 // kHz
    int mem_clock_khz        = prop.memoryClockRate;           // kHz (effective)
    int mem_bus_width_bits   = prop.memoryBusWidth;
    size_t total_global_mem  = prop.totalGlobalMem;

    int fp32_per_sm          = get_fp32_cores_per_sm(major, minor);

    // Peak FP32 GFLOPS
    double peak_fp32_gflops = -1.0;
    if (fp32_per_sm > 0 && sm_clock_khz > 0) {
        double sm_clock_hz = static_cast<double>(sm_clock_khz) * 1000.0;
        double peak_flops = 2.0 * fp32_per_sm * num_sms * sm_clock_hz; // 2 FLOPs / cycle per core
        peak_fp32_gflops = peak_flops / 1e9;
    }

    // Peak memory bandwidth GB/s (approx)
    double peak_mem_bandwidth_gbps = -1.0;
    if (mem_clock_khz > 0 && mem_bus_width_bits > 0) {
        double mem_clock_hz = static_cast<double>(mem_clock_khz) * 1000.0;
        double bw_bytes_per_s = 2.0 * mem_clock_hz * mem_bus_width_bits / 8.0; // DDR, bits->bytes
        peak_mem_bandwidth_gbps = bw_bytes_per_s / 1e9;
    }

    // Benchmarks
    double sustained_bw_gbps      = benchmark_sustained_bandwidth();
    double sustained_compute_gflops = benchmark_sustained_compute();

    // Print JSON
    printf("{\n");
    printf("  \"device_name\": \"%s\",\n", device_name);
    printf("  \"architecture_name\": \"%s\",\n", arch_name);
    printf("  \"compute_capability\": \"%d.%d\",\n", major, minor);

    printf("  \"num_sms\": %d,\n", num_sms);
    printf("  \"warp_size\": %d,\n", warp_size);
    printf("  \"max_threads_per_sm\": %d,\n", max_threads_per_sm);
    printf("  \"max_blocks_per_sm\": %d,\n", max_blocks_per_sm);
    printf("  \"registers_per_sm\": %d,\n", regs_per_sm);
    printf("  \"shared_mem_per_sm\": %d,\n", shared_mem_per_sm);
    printf("  \"l2_cache_size\": %d,\n", l2_cache_size);

    printf("  \"sm_clock_khz\": %d,\n", sm_clock_khz);
    printf("  \"mem_clock_khz\": %d,\n", mem_clock_khz);
    printf("  \"mem_bus_width_bits\": %d,\n", mem_bus_width_bits);
    printf("  \"total_global_mem\": %zu,\n", total_global_mem);

    printf("  \"peak_fp32_gflops\": %.3f,\n", peak_fp32_gflops);
    printf("  \"peak_mem_bandwidth_gbps\": %.3f,\n", peak_mem_bandwidth_gbps);
    printf("  \"sustained_bandwidth_gbps\": %.3f,\n", sustained_bw_gbps);
    printf("  \"sustained_compute_gflops\": %.3f\n", sustained_compute_gflops);
    printf("}\n");

    return 0;
}
