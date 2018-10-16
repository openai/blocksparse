
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
int GetCountSMsVersion(int* major, int* minor)
{
    CUdevice device; int count;
    cuCtxGetDevice(&device);
    cuDeviceGetAttribute(&count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
    if (major != NULL)
        cuDeviceGetAttribute(major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    if (minor != NULL)
        cuDeviceGetAttribute(minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    return count;
}

// Returns current wall time in micros.
static double NowMicros() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<unsigned long long>(tv.tv_sec) * 1000000 + tv.tv_usec;
}


Benchmark::Benchmark(CUstream stream, const char* name, float mem_size, float num_flops, int repeat, bool isgpu)
{
    stream_    = stream;
    name_      = name;
    mem_size_  = mem_size;
    num_flops_ = num_flops * 2.0f;
    repeat_    = (float)repeat;
    isgpu_     = isgpu;

    if (isgpu)
    {
        CUDA_CHECK( cuEventCreate(&hStart_, CU_EVENT_BLOCKING_SYNC) );
        CUDA_CHECK( cuEventCreate(&hStop_,  CU_EVENT_BLOCKING_SYNC) );
        CUDA_CHECK( cuEventRecord(hStart_, stream_) );
    }
    else
        us_start_ = NowMicros();
}
Benchmark::~Benchmark()
{
    float ms = 1.0f;
    if (isgpu_)
    {
        CUDA_CHECK( cuEventRecord(hStop_, stream_) );
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
        float gflops = num_flops_ / (ms * 1000000.0f);
        printf("%s fma: %12.0f ms:%8.4f gflops %5.0f\n", name_, num_flops_, ms, gflops);
        //printf("ms:%8.4f Gflops:%5.0f name: %s\n", ms, gflops, name_);
    }
}

typedef unsigned long long uint64;

// http://www.hackersdelight.org/hdcodetxt/magicgu.py.txt
void magicu64(uint d, uint &magic, uint &shift)
{
    // common cases
         if (d == 1) { magic = 1; shift = 0; }
    else if (d == 2) { magic = 1; shift = 1; }
    else if (d == 4) { magic = 1; shift = 2; }
    else if (d == 8) { magic = 1; shift = 3; }
    else
    {
        // 3 is a special case that only ends up in the high bits if the nmax is 0xffffffff
        // we can't use 0xffffffff for all cases as some return a 33 bit magic number
        uint   nbits = d == 3 ?   (2*32)+1 :   (2*31)+1;
        uint64 nmax  = d == 3 ? 0xffffffff : 0x7fffffff;
        uint64 d64   = d;
        uint64 nc    = ((nmax + 1ull) / d64) * d64 - 1ull;

        for (uint p = 0; p < nbits; p++)
        {
            if ((1ull << p) > nc * (d64 - 1ull - ((1ull << p) - 1ull) % d64))
            {
                magic = (uint)(((1ull << p) + d64 - 1ull - ((1ull << p) - 1ull) % d64) / d64);
                shift = magic == 1 ? p : p - 32;
                //printf("div:%u magic:%u shift:%u\n", d, magic, shift);
                return;
            }
        }
    }
}

// def _magic32u(nmax, d):
//     nc = ((nmax + 1) / d) * d - 1
//     nbits = len(bin(nmax)) - 2
//     for p in range(0, 2 * nbits + 1):
//         if 2 ** p > nc * (d - 1 - (2 ** p - 1) % d):
//             m = (2 ** p + d - 1 - (2 ** p - 1) % d) // d
//             return (m, p)
//     raise ValueError("Can't find magic number for division")
//
// _magic32u(0xffffffff, 3)
