
#include <cuda.h>
#include <vector_types.h>
#include <cuda_runtime.h>

// Eigen::half - IEEE half floating point memory format support (not used for compute)
// 5 bits expoenent, 10 bits mantissa, 1 bit sign
typedef struct __align__(2) ehalf {
    __device__ __forceinline__ ehalf() {}
    __device__ __forceinline__ ehalf(const unsigned short val) : x(val) {}
    unsigned short x;
} ehalf;

typedef struct __align__(4) ehalf2 {
    __device__ __forceinline__ ehalf2() {}
    __device__ __forceinline__ ehalf2(const unsigned int val) : x(val) {}
    unsigned int x;
} ehalf2;

typedef struct __align__(8) ehalf4 {
    __device__ __forceinline__ ehalf4() {}
    __device__ __forceinline__ ehalf4(const unsigned int val) : x(val), y(val) {}
    unsigned int x;
    unsigned int y;
} ehalf4;

typedef struct __align__(16) ehalf8 {
    __device__ __forceinline__ ehalf8() {}
    __device__ __forceinline__ ehalf8(const unsigned int val) : x(val), y(val), z(val), w(val) {}
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
} ehalf8;


// tf.bfloat16 half floating point memory format support (not used for compute)
// 8 bits expoenent, 7 bits mantissa, 1 bit sign

typedef struct __align__(2) bhalf {
    __device__ __forceinline__ bhalf() {}
    __device__ __forceinline__ bhalf(const unsigned short val) : x(val) {}
    unsigned short x;
} bhalf;

typedef struct __align__(4) bhalf2 {
    __device__ __forceinline__ bhalf2() {}
    __device__ __forceinline__ bhalf2(const unsigned int val) : x(val) {}
    unsigned int x;
} bhalf2;

typedef struct __align__(8) bhalf4 {
    __device__ __forceinline__ bhalf4() {}
    __device__ __forceinline__ bhalf4(const unsigned int val) : x(val), y(val) {}
    unsigned int x;
    unsigned int y;
} bhalf4;

typedef struct __align__(16) bhalf8 {
    __device__ __forceinline__ bhalf8() {}
    __device__ __forceinline__ bhalf8(const unsigned int val) : x(val), y(val), z(val), w(val) {}
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
} bhalf8;


typedef struct __align__(16) float8 {
    __device__ __forceinline__ float8() { }
    //__device__ __forceinline__ float8(const float val) : a({val,val,val,val}), b({val,val,val,val}) { }
    float4 a;
    float4 b;
} float8;


// simple aliases to help keep nice and aligned
#define FLOAT float
#define EHALF Eigen::half
#define BHALF bfloat16

// Map TF types to GPU single and vector types
#define FLOAT2 float,float,float2,float4,float8
#define EHALF2 Eigen::half,ehalf,ehalf2,ehalf4,ehalf8
#define BHALF2 bfloat16,bhalf,bhalf2,bhalf4,bhalf8



// Mixed Types - for TF op template declaratoins
#define MTYPE1(ta) typename ta, typename ta##1, typename ta##2, typename ta##4, typename ta##8
#define MTYPE2(ta,tb)    MTYPE1(ta), MTYPE1(tb)
#define MTYPE3(ta,tb,tc) MTYPE1(ta), MTYPE1(tb), MTYPE1(tc)

#define NTYPE1(a) a, a##1, a##2, a##4, a##8
#define NTYPE2(a, b)    NTYPE1(a), NTYPE1(b)
#define NTYPE3(a, b, c) NTYPE1(a), NTYPE1(b), NTYPE1(c)

#define OTYPE1(a) a##1, a##2, a##4, a##8
#define OTYPE2(a, b)    OTYPE1(a), OTYPE1(b)
#define OTYPE3(a, b, c) OTYPE1(a), OTYPE1(b), OTYPE1(c)



// Cuda Types - for cuda template declaratoins
#define CTYPE1(ta) typename ta, typename ta##2, typename ta##4, typename ta##8
#define CTYPE2(ta,tb)    CTYPE1(ta), CTYPE1(tb)
#define CTYPE3(ta,tb,tc) CTYPE1(ta), CTYPE1(tb), CTYPE1(tc)

// Vector Types - for cuda implementation calling/instantiation
#define VTYPE1(a) a, a##2, a##4, a##8
#define VTYPE2(a, b)    VTYPE1(a), VTYPE1(b)
#define VTYPE3(a, b, c) VTYPE1(a), VTYPE1(b), VTYPE1(c)


template<typename T>
struct plist8 {
    const T* a[8];
};

typedef struct bsmm_params
{
    const int* Lut;
    int* Lock;
    //float4* Scratch;
    int blocks;
    int bshift;
    int segments;
    int locks;
    int C;
    int K;
    int N;
    int shared;
    int pcount;
    float alpha;
    float beta;
    CUstream stream;
} bsmm_params;


int GetCountSMs();

class Benchmark
{
  public:
    Benchmark(const char* name, float mem_size, float num_flops, int repeat, bool isgpu=true);
    ~Benchmark();
    const char* name_;
    float mem_size_, num_flops_, repeat_;
    CUevent hStart_, hStop_;
    bool isgpu_;
    double us_start_;
};