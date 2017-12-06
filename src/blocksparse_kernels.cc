
#include "tensorflow/core/framework/op.h"
#include <stdio.h>
#include <cuda.h>
#include <utility>
#include <string>
#include <map>
#include <mutex>

#include "blocksparse_kernels.h"

using namespace tensorflow;

std::map<std::pair<CUcontext, std::string>, CUfunction> kernels_;
std::mutex kernels_mutex_;

#define CUDA_CHECK( fn ) do { \
    CUresult status = (fn); \
    if ( CUDA_SUCCESS != status ) { \
      const char* errstr; \
      cuGetErrorString(status, &errstr); \
      return errors::Internal(errstr); \
    } \
  } while (0)

Status GetKernel(std::string& kernel_name, CUfunction* kernel)
{
  // Only need to get kernel once.
  if (*kernel)
    return Status::OK();

  CUcontext context;
  CUDA_CHECK( cuCtxGetCurrent(&context) );

  auto key = std::make_pair(context, kernel_name);

  std::lock_guard<std::mutex> lock(kernels_mutex_);

  auto kernel_pair = kernels_.find(key);
  if (kernel_pair != kernels_.end())
  {
    *kernel = kernel_pair->second;
    //printf("found:  %s\n", kernel_name.c_str());
  }
  else
  {
    CUdevice device;
    CUmodule module;
    int major;

    auto kernel_data_pair = kernel_map_.find(kernel_name);
    if (kernel_data_pair == kernel_map_.end())
    {
      std::string errstr = kernel_name + " not availble.";
      return errors::Internal(errstr.c_str());
    }

    const uint8_t* kernel_data_src = kernel_data_pair->second.first;
    size_t kernel_data_size        = kernel_data_pair->second.second;

    uint8_t* kernel_data = (uint8_t*)malloc(kernel_data_size);
    memcpy(kernel_data, kernel_data_src, kernel_data_size);

    CUDA_CHECK( cuCtxGetDevice(&device) );
    CUDA_CHECK( cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) );

    if (major == 6)
    {
      // SM 50 and 60 cubins are identical except for these bytes in the elf header flags
      kernel_data[48] = 60;
      kernel_data[49] = 13;
      kernel_data[50] = 60;
    }
    CUDA_CHECK( cuModuleLoadData(&module, kernel_data) );
    CUDA_CHECK( cuModuleGetFunction(kernel, module, kernel_name.c_str()) );

    free(kernel_data);

    kernels_.insert(std::make_pair(key, *kernel));
    //printf("insert: %s\n", kernel_name.c_str());

  }
  return Status::OK();
}



