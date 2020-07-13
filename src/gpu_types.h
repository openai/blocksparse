
#ifndef GPU_TYPES_H
#define GPU_TYPES_H

#include <cuda.h>
#include <vector_types.h>
#include <cuda_runtime.h>

#ifndef FLT_MAX
#define FLT_MAX 3.402823466E+38F
#endif // FLT_MAX

// Eigen::half - IEEE half floating point memory format support (not used for compute)
// 5 bits expoenent, 10 bits mantissa, 1 bit sign
typedef struct __align__(2) ehalf {
    __device__ __forceinline__ ehalf() {}
    __device__ __forceinline__ ehalf(ushort v) : x(v) {}
    ushort x;
} ehalf;

typedef struct __align__(4) ehalf2 {
    __device__ __forceinline__ ehalf2() {}
    __device__ __forceinline__ ehalf2(uint v) : x(v) {}
    uint x;
} ehalf2;

typedef struct __align__(8) ehalf4 {
    __device__ __forceinline__ ehalf4() {}
    __device__ __forceinline__ ehalf4(uint v) : x(v), y(v) {}
    __device__ __forceinline__ ehalf4(uint v0, uint v1) : x(v0), y(v1) {}
    uint x;
    uint y;
} ehalf4;

typedef struct __align__(16) ehalf8 {
    __device__ __forceinline__ ehalf8() {}
    __device__ __forceinline__ ehalf8(uint v) : x(v), y(v), z(v), w(v) {}
    __device__ __forceinline__ ehalf8(uint v0, uint v1, uint v2, uint v3) : x(v0), y(v1), z(v2), w(v3) {}
    uint x;
    uint y;
    uint z;
    uint w;
} ehalf8;


// tf.bfloat16 half floating point memory format support (not used for compute)
// 8 bits expoenent, 7 bits mantissa, 1 bit sign
typedef struct __align__(2) bhalf {
    __device__ __forceinline__ bhalf() {}
    __device__ __forceinline__ bhalf(ushort v) : x(v) {}
    ushort x;
} bhalf;

typedef struct __align__(4) bhalf2 {
    __device__ __forceinline__ bhalf2() {}
    __device__ __forceinline__ bhalf2(uint v) : x(v) {}
    uint x;
} bhalf2;

typedef struct __align__(8) bhalf4 {
    __device__ __forceinline__ bhalf4() {}
    __device__ __forceinline__ bhalf4(uint v) : x(v), y(v) {}
    __device__ __forceinline__ bhalf4(uint v0, uint v1) : x(v0), y(v1) {}
    uint x;
    uint y;
} bhalf4;

typedef struct __align__(16) bhalf8 {
    __device__ __forceinline__ bhalf8() {}
    __device__ __forceinline__ bhalf8(uint v) : x(v), y(v), z(v), w(v) {}
    __device__ __forceinline__ bhalf8(uint v0, uint v1, uint v2, uint v3) : x(v0), y(v1), z(v2), w(v3) {}
    uint x;
    uint y;
    uint z;
    uint w;
} bhalf8;


// vhalf: 16 bit storage format for Adam running gradient variance (unsigned)
// 6 bits expoenent, 10 bits mantissa, 0 bit sign
typedef struct __align__(2) vhalf {
    __device__ __forceinline__ vhalf() {}
    __device__ __forceinline__ vhalf(ushort v) : x(v) {}
    ushort x;
} vhalf;

typedef struct __align__(4) vhalf2 {
    __device__ __forceinline__ vhalf2() {}
    __device__ __forceinline__ vhalf2(uint v) : x(v) {}
    uint x;
} vhalf2;

typedef struct __align__(8) vhalf4 {
    __device__ __forceinline__ vhalf4() {}
    __device__ __forceinline__ vhalf4(uint v) : x(v), y(v) {}
    __device__ __forceinline__ vhalf4(uint v0, uint v1) : x(v0), y(v1) {}
    uint x;
    uint y;
} vhalf4;


// mhalf: 16 bit storage format for Adam running gradient mean (signed)
// 6 bits expoenent, 9 bits mantissa, 1 bit sign
typedef struct __align__(2) mhalf {
    __device__ __forceinline__ mhalf() {}
    __device__ __forceinline__ mhalf(ushort v) : x(v) {}
    ushort x;
} mhalf;

typedef struct __align__(4) mhalf2 {
    __device__ __forceinline__ mhalf2() {}
    __device__ __forceinline__ mhalf2(uint v) : x(v) {}
    uint x;
} mhalf2;

typedef struct __align__(8) mhalf4 {
    __device__ __forceinline__ mhalf4() {}
    __device__ __forceinline__ mhalf4(uint v) : x(v), y(v) {}
    __device__ __forceinline__ mhalf4(uint v0, uint v1) : x(v0), y(v1) {}
    uint x;
    uint y;
} mhalf4;


typedef struct __align__(16) float8 {
    __device__ __forceinline__ float8() { }
    __device__ __forceinline__ float8(float4 x, float4 y) : a(x), b(y) {}
    //__device__ __forceinline__ float8(const float val) : a({val,val,val,val}), b({val,val,val,val}) { }
    float4 a;
    float4 b;
} float8;

#define CTX_STREAM(ctx) reinterpret_cast<CUstream>(ctx->op_device_context()->stream()->implementation()->GpuStreamHack())

#define NT_OP 0
#define NN_OP 1
#define TN_OP 2

// simple aliases to help keep nice and aligned
#define FLOAT float
#define EHALF Eigen::half
#define MHALF Eigen::half
#define VHALF Eigen::half
#define BHALF bfloat16

// Map TF types to GPU single and vector types
#define FLOAT_V float,      float,float2,float4,float8
#define EHALF_V Eigen::half,ehalf,ehalf2,ehalf4,ehalf8
#define BHALF_V bfloat16,   bhalf,bhalf2,bhalf4,bhalf8


// Mixed Types - for mapping TF types to our vector types in template declarations
#define MTYPE(a) typename a, typename a##1, typename a##2, typename a##4, typename a##8

// Cuda Types - for cuda template declaratoins
#define CTYPE(a) typename a, typename a##2, typename a##4, typename a##8

// Vector Types - for cuda implementation calling/instantiation
#define NTYPE(a) a##1, a##2, a##4, a##8
#define VTYPE(a) a, a##2, a##4, a##8

template<uint DIMS>
struct Strides {
    uint stride[DIMS];
};

template<typename T, uint SIZE>
struct Plist {
    const T* a[SIZE];
};

typedef struct bsmm_params
{
    const int* Lut;
    const float* Gate;
    int* Lock;
    //float4* Scratch;
    int blocks;
    int bsize;
    int segments;
    int locks;
    int C;
    int K;
    int N;
    int shared;
    int pcount;
    uint blk_a;
    uint blk_A;
    uint blk_b;
    uint blk_B;
    float alpha;
    float beta;
    CUstream stream;
} bsmm_params;


int GetCountSMs();
int GetCountSMsVersion(int* major, int* minor);

class Benchmark
{
  public:
    Benchmark(CUstream stream, const char* name, float mem_size, float num_flops, int repeat, bool isgpu=true);
    ~Benchmark();
    CUstream stream_;
    const char* name_;
    float mem_size_, num_flops_, repeat_;
    CUevent hStart_, hStop_;
    bool isgpu_;
    double us_start_;
};

#define CEIL_DIV(x, y) (((x) + (y) -   1) / (y))
#define LOG2(x)        (x==1?0: x==2?1: x<=4?2: x<=8?3: x<=16?4: x<=32?5: x<=64?6: x<=128?7: x<=256?8: x<=512?9 : x<=1024?10 : x<=2048?11 : 12)
#define THREAD_POW2(x) (x<=32?32: x<=64?64: x<=128?128: x<=256?256: x<=512?512: 1024)

typedef struct QuantStats
{
    float mean;
    float stdv;
    float sat_pct;
    float ftz_pct;
    float max_val;
} QuantStats;

void magicu64(uint d, uint &magic, uint &shift);

#endif // GPU_TYPES_H