// runner/main.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// === kernel headers (must exist in ../kernels) ===
#include "../kernels/vector_add.cuh"
#include "../kernels/saxpy.cuh"
#include "../kernels/strided_copy_8.cuh"
#include "../kernels/naive_transpose.cuh"
#include "../kernels/matmul_tiled.cuh"
#include "../kernels/reduce_sum.cuh"
#include "../kernels/dot_product.cuh"
#include "../kernels/histogram.cuh"
#include "../kernels/conv2d_3x3.cuh"
#include "../kernels/conv2d_7x7.cuh"
#include "../kernels/shared_transpose.cuh"
#include "../kernels/random_access.cuh"
#include "../kernels/vector_add_divergent.cuh"
#include "../kernels/shared_bank_conflict.cuh"
#include "../kernels/matmul_naive.cuh"
#include "../kernels/atomic_hotspot.cuh"

// ---------- helpers ----------
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,             \
              cudaGetErrorString(err__));                                       \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

template <typename F>
float time_ms(F launch, int warm, int reps) {
  // warmup
  for (int i = 0; i < warm; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < reps; ++i) launch();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return ms / reps;
}

int main(int argc, char** argv) {
  // defaults
  std::string kernel = "vector_add";
  int N = 1 << 20;            // used by 1D kernels
  int block = 256;
  int warm = 20, reps = 100;
  int rows = 2048, cols = 2048; // transpose
  int matN = 512;               // matmul sizes
  float alpha = 2.0f;           // saxpy
  int H = 1024, W = 1024;       // conv
  int iters = 100;              // atomic_hotspot

  // arg parsing
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--kernel") && i + 1 < argc) kernel = argv[++i];
    else if (!strcmp(argv[i], "--N") && i + 1 < argc) N = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--block") && i + 1 < argc) block = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warm = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--rows") && i + 1 < argc) rows = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--cols") && i + 1 < argc) cols = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--matN") && i + 1 < argc) matN = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--alpha") && i + 1 < argc) alpha = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--H") && i + 1 < argc) H = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--W") && i + 1 < argc) W = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
  }

  // ===== kernels =====
  if (kernel == "vector_add") {
    size_t bytes = (size_t)N * sizeof(float);
    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMalloc(&C, bytes));
    CUDA_CHECK(cudaMemset(A, 0, bytes));
    CUDA_CHECK(cudaMemset(B, 0, bytes));
    dim3 blk(block), grid((N + block - 1) / block);
    auto launch = [&]() { vector_add_kernel<<<grid, blk>>>(A, B, C, N); };
    float ms = time_ms(launch, warm, reps);
    printf("KERNEL=vector_add N=%d block=%d grid=%d time_ms=%.6f\n",
           N, block, grid.x, ms);
    cudaFree(A); cudaFree(B); cudaFree(C); return 0;
  }

  if (kernel == "saxpy") {
    size_t bytes = (size_t)N * sizeof(float);
    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMalloc(&C, bytes));
    CUDA_CHECK(cudaMemset(A, 0, bytes));
    CUDA_CHECK(cudaMemset(B, 0, bytes));
    dim3 blk(block), grid((N + block - 1) / block);
    auto launch = [&]() { saxpy_kernel<<<grid, blk>>>(alpha, A, B, C, N); };
    float ms = time_ms(launch, warm, reps);
    printf("KERNEL=saxpy N=%d block=%d grid=%d time_ms=%.6f\n",
           N, block, grid.x, ms);
    cudaFree(A); cudaFree(B); cudaFree(C); return 0;
  }

  if (kernel == "strided_copy_8") {
    size_t bytes = (size_t)N * sizeof(float);
    float *A, *C;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&C, bytes));
    CUDA_CHECK(cudaMemset(A, 0, bytes));
    CUDA_CHECK(cudaMemset(C, 0, bytes));
    // indices processed = ceil(N/8)
    int nidx = (N + 7) / 8;
    dim3 blk(block), grid((nidx + block - 1) / block);
    auto launch = [&]() { strided_copy_8_kernel<<<grid, blk>>>(A, C, N); };
    float ms = time_ms(launch, warm, reps);
    printf("KERNEL=strided_copy_8 N=%d block=%d grid=%d time_ms=%.6f\n",
           N, block, grid.x, ms);
    cudaFree(A); cudaFree(C); return 0;
  }

  if (kernel == "naive_transpose") {
    size_t bytes = (size_t)rows * cols * sizeof(float);
    float *A, *B;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMemset(A, 0, bytes));
    CUDA_CHECK(cudaMemset(B, 0, bytes));
    dim3 blk(16,16), grid((cols + 15)/16, (rows + 15)/16);
    auto launch = [&]() { naive_transpose_kernel<<<grid, blk>>>(A, B, rows, cols); };
    float ms = time_ms(launch, warm, reps);
    printf("KERNEL=naive_transpose rows=%d cols=%d block=(16,16) grid=(%d,%d) time_ms=%.6f\n",
           rows, cols, grid.x, grid.y, ms);
    cudaFree(A); cudaFree(B); return 0;
  }

  if (kernel == "shared_transpose") {
    size_t bytes = (size_t)rows * cols * sizeof(float);
    float *A, *B;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMemset(A, 0, bytes));
    CUDA_CHECK(cudaMemset(B, 0, bytes));
    dim3 blk(32,32), grid((cols + 31)/32, (rows + 31)/32);
    auto launch = [&]() { shared_transpose_kernel<<<grid, blk>>>(A, B, rows, cols); };
    float ms = time_ms(launch, warm, reps);
    printf("KERNEL=shared_transpose rows=%d cols=%d block=(32,32) grid=(%d,%d) time_ms=%.6f\n",
           rows, cols, grid.x, grid.y, ms);
    cudaFree(A); cudaFree(B); return 0;
  }

  if (kernel == "matmul_tiled") {
    int N_ = matN;
    size_t bytes = (size_t)N_ * N_ * sizeof(float);
    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMalloc(&C, bytes));
    CUDA_CHECK(cudaMemset(A, 0, bytes));
    CUDA_CHECK(cudaMemset(B, 0, bytes));
#ifndef TILE
#define TILE 32
#endif
    dim3 blk(TILE, TILE), grid((N_ + TILE - 1)/TILE, (N_ + TILE - 1)/TILE);
    auto launch = [&]() { matmul_tiled_kernel<<<grid, blk>>>(A, B, C, N_); };
    float ms = time_ms(launch, warm, reps);
    printf("KERNEL=matmul_tiled matN=%d TILE=%d block=(%d,%d) grid=(%d,%d) time_ms=%.6f\n",
           N_, TILE, blk.x, blk.y, grid.x, grid.y, ms);
    cudaFree(A); cudaFree(B); cudaFree(C); return 0;
  }

  if (kernel == "matmul_naive") {
    int N_ = matN;
    size_t bytes = (size_t)N_ * N_ * sizeof(float);
    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMalloc(&C, bytes));
    CUDA_CHECK(cudaMemset(A, 0, bytes));
    CUDA_CHECK(cudaMemset(B, 0, bytes));
    dim3 blk(16,16), grid((N_ + 15)/16, (N_ + 15)/16);
    auto launch = [&]() { matmul_naive_kernel<<<grid, blk>>>(A, B, C, N_); };
    float ms = time_ms(launch, warm, reps);
    printf("KERNEL=matmul_naive matN=%d block=(%d,%d) grid=(%d,%d) time_ms=%.6f\n",
           N_, blk.x, blk.y, grid.x, grid.y, ms);
    cudaFree(A); cudaFree(B); cudaFree(C); return 0;
  }

  if (kernel == "reduce_sum") {
    size_t bytes = (size_t)N * sizeof(float);
    float *A, *partials;
    int blk = block, grid = (N + blk*2 - 1) / (blk*2);
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&partials, grid * sizeof(float)));
    CUDA_CHECK(cudaMemset(A, 0, bytes));
    CUDA_CHECK(cudaMemset(partials, 0, grid * sizeof(float)));
    auto launch = [&]() { reduce_sum_kernel<<<grid, blk, blk * sizeof(float)>>>(A, partials, N); };
    float ms = time_ms(launch, warm, reps);
    printf("KERNEL=reduce_sum N=%d block=%d grid=%d time_ms=%.6f\n",
           N, blk, grid, ms);
    cudaFree(A); cudaFree(partials); return 0;
  }

  if (kernel == "dot_product") {
    size_t bytes = (size_t)N * sizeof(float);
    float *A, *B, *partials;
    int blk = block, grid = (N + blk*2 - 1) / (blk*2);
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMalloc(&partials, grid * sizeof(float)));
    CUDA_CHECK(cudaMemset(A, 0, bytes));
    CUDA_CHECK(cudaMemset(B, 0, bytes));
    CUDA_CHECK(cudaMemset(partials, 0, grid * sizeof(float)));
    auto launch = [&]() { dot_product_kernel<<<grid, blk, blk * sizeof(float)>>>(A, B, partials, N); };
    float ms = time_ms(launch, warm, reps);
    printf("KERNEL=dot_product N=%d block=%d grid=%d time_ms=%.6f\n",
           N, blk, grid, ms);
    cudaFree(A); cudaFree(B); cudaFree(partials); return 0;
  }

  if (kernel == "histogram") {
    size_t bytes = (size_t)N * sizeof(unsigned int);
    unsigned int *data, *bins;
    CUDA_CHECK(cudaMalloc(&data, bytes));
    CUDA_CHECK(cudaMalloc(&bins, 256 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(data, 0, bytes));
    CUDA_CHECK(cudaMemset(bins, 0, 256 * sizeof(unsigned int)));
    int blk = block, grid = (N + blk - 1) / blk;
    auto launch = [&]() { histogram_kernel<<<grid, blk, 256 * sizeof(unsigned int)>>>(data, N, bins); };
    float ms = time_ms(launch, warm, reps);
    printf("KERNEL=histogram N=%d block=%d grid=%d time_ms=%.6f\n",
           N, blk, grid, ms);
    cudaFree(data); cudaFree(bins); return 0;
  }

  if (kernel == "conv2d_3x3") {
    size_t bytes = (size_t)H * W * sizeof(float);
    float *img, *k, *out;
    CUDA_CHECK(cudaMalloc(&img, bytes));
    CUDA_CHECK(cudaMalloc(&out, bytes));
    CUDA_CHECK(cudaMalloc(&k, 9 * sizeof(float)));
    CUDA_CHECK(cudaMemset(img, 0, bytes));
    CUDA_CHECK(cudaMemset(out, 0, bytes));
    CUDA_CHECK(cudaMemset(k, 0, 9 * sizeof(float)));
    dim3 blk(16,16), grid((W + 15)/16, (H + 15)/16);
    auto launch = [&]() { conv2d_3x3_kernel<<<grid, blk>>>(img, k, out, H, W); };
    float ms = time_ms(launch, warm, reps);
    printf("KERNEL=conv2d_3x3 H=%d W=%d block=(16,16) grid=(%d,%d) time_ms=%.6f\n",
           H, W, grid.x, grid.y, ms);
    cudaFree(img); cudaFree(out); cudaFree(k); return 0;
  }

  if (kernel == "conv2d_7x7") {
    size_t bytes = (size_t)H * W * sizeof(float);
    float *img, *k, *out;
    CUDA_CHECK(cudaMalloc(&img, bytes));
    CUDA_CHECK(cudaMalloc(&out, bytes));
    CUDA_CHECK(cudaMalloc(&k, 49 * sizeof(float)));
    CUDA_CHECK(cudaMemset(img, 0, bytes));
    CUDA_CHECK(cudaMemset(out, 0, bytes));
    CUDA_CHECK(cudaMemset(k, 0, 49 * sizeof(float)));
    dim3 blk(16,16), grid((W + 15)/16, (H + 15)/16);
    auto launch = [&]() { conv2d_7x7_kernel<<<grid, blk>>>(img, k, out, H, W); };
    float ms = time_ms(launch, warm, reps);
    printf("KERNEL=conv2d_7x7 H=%d W=%d block=(16,16) grid=(%d,%d) time_ms=%.6f\n",
           H, W, grid.x, grid.y, ms);
    cudaFree(img); cudaFree(out); cudaFree(k); return 0;
  }

  if (kernel == "random_access") {
    size_t bytes = (size_t)N * sizeof(float);
    size_t ibytes = (size_t)N * sizeof(int);
    float *A, *B; int *idx;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMalloc(&idx, ibytes));
    CUDA_CHECK(cudaMemset(A, 0, bytes));
    CUDA_CHECK(cudaMemset(B, 0, bytes));
    CUDA_CHECK(cudaMemset(idx, 0, ibytes));
    int blk = block, grid = (N + blk - 1) / blk;
    auto launch = [&]() { random_access_kernel<<<grid, blk>>>(A, idx, B, N); };
    float ms = time_ms(launch, warm, reps);
    printf("KERNEL=random_access N=%d block=%d grid=%d time_ms=%.6f\n",
           N, blk, grid, ms);
    cudaFree(A); cudaFree(B); cudaFree(idx); return 0;
  }

  if (kernel == "vector_add_divergent") {
    size_t bytes = (size_t)N * sizeof(float);
    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMalloc(&C, bytes));
    CUDA_CHECK(cudaMemset(A, 0, bytes));
    CUDA_CHECK(cudaMemset(B, 0, bytes));
    int blk = block, grid = (N + blk - 1) / blk;
    auto launch = [&]() { vector_add_divergent_kernel<<<grid, blk>>>(A, B, C, N); };
    float ms = time_ms(launch, warm, reps);
    printf("KERNEL=vector_add_divergent N=%d block=%d grid=%d time_ms=%.6f\n",
           N, blk, grid, ms);
    cudaFree(A); cudaFree(B); cudaFree(C); return 0;
  }

  if (kernel == "shared_bank_conflict") {
    float *out;
    CUDA_CHECK(cudaMalloc(&out, 1024 * sizeof(float)));
    dim3 blk(1024), grid(1);
    auto launch = [&]() { shared_bank_conflict_kernel<<<grid, blk>>>(out); };
    float ms = time_ms(launch, warm, reps);
    printf("KERNEL=shared_bank_conflict block=(1024) grid=(1) regs=206 shmem=4096 time_ms=%.6f\n", ms);
    cudaFree(out); return 0;
  }

  if (kernel == "atomic_hotspot") {
    unsigned int *ctr;
    CUDA_CHECK(cudaMalloc(&ctr, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(ctr, 0, sizeof(unsigned int)));
    int blk = block, grid = (N + blk - 1) / blk;
    auto launch = [&]() { atomic_hotspot_kernel<<<grid, blk>>>(ctr, iters); };
    float ms = time_ms(launch, warm, reps);
    printf("KERNEL=atomic_hotspot N=%d block=%d grid=%d iters=%d time_ms=%.6f\n",
           N, blk, grid, iters, ms);
    cudaFree(ctr); return 0;
  }

  fprintf(stderr, "Unknown --kernel\n");
  return 1;
}

