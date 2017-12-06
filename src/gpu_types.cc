
#include "gpu_types.h"
#include <stdio.h>
#include <sys/time.h>


#define CUDA_CHECK( fn ) do { \
    CUresult status = (fn); \
    if ( CUDA_SUCCESS != status ) { \
        const char* errstr; \
        cuGetErrorString(status, &errstr); \
        printf("CUDA Driver Failure (line %d of file %s):\n\t%s returned 0x%x (%s)\n", __LINE__, __FILE__, #fn, status, errstr); \
    } \
} while (0)


int GetCountSMs()
{
    CUdevice device; int count;
    cuCtxGetDevice(&device);
    cuDeviceGetAttribute(&count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
    return count;
}

// Returns current wall time in micros.
static double NowMicros() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<unsigned long long>(tv.tv_sec) * 1000000 + tv.tv_usec;
}


Benchmark::Benchmark(const char* name, float mem_size, float num_flops, int repeat, bool isgpu)
{
    name_      = name;
    mem_size_  = mem_size;
    num_flops_ = num_flops;
    repeat_    = (float)repeat;
    isgpu_     = isgpu;

    if (isgpu)
    {
        CUDA_CHECK( cuEventCreate(&hStart_, CU_EVENT_BLOCKING_SYNC) );
        CUDA_CHECK( cuEventCreate(&hStop_,  CU_EVENT_BLOCKING_SYNC) );
        CUDA_CHECK( cuEventRecord(hStart_, NULL) );
    }
    else
        us_start_ = NowMicros();
}
Benchmark::~Benchmark()
{
    float ms = 1.0f;
    if (isgpu_)
    {
        CUDA_CHECK( cuEventRecord(hStop_, NULL) );
        CUDA_CHECK( cuEventSynchronize(hStop_) );
        CUDA_CHECK( cuEventElapsedTime(&ms, hStart_, hStop_) );
        CUDA_CHECK( cuEventDestroy(hStart_) );
        CUDA_CHECK( cuEventDestroy(hStop_) );
    }
    else
        ms = (float)(NowMicros() - us_start_) / 1000.0f;

    ms /= repeat_;
    if (mem_size_ != 0.0f)
    {
        float gbps   = mem_size_ / (ms * 1024.0f*1024.0f);
        printf("ms:%8.4f GBps:%4.0f name: %s\n", ms, gbps, name_);
    }
    else
    {
        float gflops = (num_flops_ * 2.0f) / (ms * 1000000.0f);
        printf("%s %4.0f\n", name_, gflops);
        //printf("ms:%8.4f Gflops:%5.0f name: %s\n", ms, gflops, name_);
    }
}
