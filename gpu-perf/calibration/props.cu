#include <cuda_runtime.h>
#include <cstdio>
int main(){
  int dev=0; cudaSetDevice(dev);
  cudaDeviceProp p{}; cudaGetDeviceProperties(&p, dev);
  printf("name=%s\n", p.name);
  printf("major=%d minor=%d\n", p.major, p.minor);
  printf("multiProcessorCount=%d\n", p.multiProcessorCount);
  printf("maxThreadsPerMultiProcessor=%d\n", p.maxThreadsPerMultiProcessor);
  printf("maxThreadsPerBlock=%d\n", p.maxThreadsPerBlock);
  printf("regsPerMultiprocessor=%d\n", p.regsPerMultiprocessor);
  printf("sharedMemPerMultiprocessor=%d\n", (int)p.sharedMemPerMultiprocessor);
  printf("sharedMemPerBlockOptin=%d\n", (int)p.sharedMemPerBlockOptin);
  printf("maxBlocksPerMultiProcessor=%d\n", p.maxBlocksPerMultiProcessor);
  printf("warpSize=%d\n", p.warpSize);
  printf("l2CacheSizeBytes=%d\n", (int)p.l2CacheSize);
  return 0;
}

