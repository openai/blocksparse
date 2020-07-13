TARGET=./build

.PHONY: all compile clean
all: compile

compile: blocksparse/blocksparse_ops.so
	python setup.py bdist_wheel --universal

clean:
	rm -vfr $(TARGET)

release: compile
	BRANCH=$(shell git rev-parse --abbrev-ref HEAD); if [ "$$BRANCH" != "master" ]; then echo "--- ERROR: refusing to build non-master branch"; exit 1; fi
	@git diff-index --quiet HEAD -- || ( echo '--- ERROR: will not build while git tree is dirty! please commit your changes. ---' && exit 1 )
	# hacky way to get the version from wheel name
	$(eval VERSION := $(shell ls -th dist/*.whl | head -1 |awk '{split($$1,r,"-");print r[2]}')) # '
	git tag v${VERSION}
	git push origin v${VERSION}

	# Upload the binary wheel to PyPi. Needs `twine` installed and configured with your PyPi credentials.
	twine upload $(shell ls -th dist/*.whl | head -1)

CUDA_HOME?=/usr/local/cuda
NV_INC?=$(CUDA_HOME)/include
NV_LIB?=$(CUDA_HOME)/lib64

NCCL_HOME?=/usr/local/nccl
NCCL_INC?=$(NCCL_HOME)/include
NCCL_LIB?=$(NCCL_HOME)/lib

MPI_HOME?=/usr/lib/mpich
MPI_INC?=$(MPI_HOME)/include
MPI_LIB?=$(MPI_HOME)/lib

TF_INC=$(shell python -c 'from os.path import dirname; import tensorflow as tf; print(dirname(dirname(tf.sysconfig.get_include())))')
TF_LIB=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_ABI=$(shell python -c 'import tensorflow as tf; print(tf.__cxx11_abi_flag__ if "__cxx11_abi_flag__" in tf.__dict__ else 0)')

CCFLAGS=-std=c++11 -O3 -fPIC -DGOOGLE_CUDA=1 -D_GLIBCXX_USE_CXX11_ABI=$(TF_ABI) \
	-I$(TARGET) \
	-I$(NV_INC) \
	-I$(TF_INC)/tensorflow/include \
	-I$(TF_INC)/tensorflow/include/external/nsync/public \
	-I$(TF_INC)/external/local_config_cuda/cuda \
	-I$(NCCL_INC) \
	-I$(MPI_INC) \
	-I/usr/local

NVCCFLAGS=-DGOOGLE_CUDA=1 -D_GLIBCXX_USE_CXX11_ABI=$(TF_ABI) -O3 -Xcompiler -fPIC -std=c++11 --prec-div=false --prec-sqrt=false \
 	-gencode=arch=compute_35,code=sm_35 \
	-gencode=arch=compute_50,code=sm_50 \
	-gencode=arch=compute_52,code=sm_52 \
 	-gencode=arch=compute_60,code=sm_60 \
	-gencode=arch=compute_61,code=sm_61 \
 	-gencode=arch=compute_70,code=sm_70 \
 	-gencode=arch=compute_70,code=compute_70
#   --keep --keep-dir tmp

OBJS=\
	$(TARGET)/batch_norm_op.o \
	$(TARGET)/blocksparse_conv_op.o \
	$(TARGET)/blocksparse_kernels.o \
	$(TARGET)/blocksparse_l2_norm_op.o \
	$(TARGET)/blocksparse_matmul_op.o \
	$(TARGET)/bst_op.o \
	$(TARGET)/cwise_linear_op.o \
	$(TARGET)/edge_bias_op.o \
	$(TARGET)/ew_op.o \
	$(TARGET)/gpu_types.o \
	$(TARGET)/layer_norm_op.o \
	$(TARGET)/lstm_op.o \
	$(TARGET)/optimize_op.o \
	$(TARGET)/quantize_op.o \
	$(TARGET)/transformer_op.o \
	$(TARGET)/embedding_op.o \
	$(TARGET)/matmul_op.o \
	$(TARGET)/nccl_op.o

CU_OBJS=\
	$(TARGET)/batch_norm_op_gpu.cu.o \
	$(TARGET)/blocksparse_l2_norm_op_gpu.cu.o \
	$(TARGET)/blocksparse_matmul_op_gpu.cu.o \
	$(TARGET)/blocksparse_hgemm_cn_64_op_gpu.cu.o \
	$(TARGET)/blocksparse_hgemm_cn_128_op_gpu.cu.o \
	$(TARGET)/blocksparse_hgemm_nc_op_gpu.cu.o \
	$(TARGET)/bst_hgemm_op_gpu.cu.o \
	$(TARGET)/bst_sgemm_op_gpu.cu.o \
	$(TARGET)/bst_softmax_op_gpu.cu.o \
	$(TARGET)/cwise_linear_op_gpu.cu.o \
	$(TARGET)/edge_bias_op_gpu.cu.o \
	$(TARGET)/ew_op_gpu.cu.o \
	$(TARGET)/layer_norm_cn_op_gpu.cu.o \
	$(TARGET)/layer_norm_nc_op_gpu.cu.o \
	$(TARGET)/lstm_op_gpu.cu.o \
	$(TARGET)/optimize_op_gpu.cu.o \
	$(TARGET)/quantize_op_gpu.cu.o \
	$(TARGET)/transformer_op_gpu.cu.o \
	$(TARGET)/embedding_op_gpu.cu.o \
	$(TARGET)/matmul_op_gpu.cu.o

$(TARGET)/blocksparse_kernels.h: src/sass/*.sass
	mkdir -p $(shell dirname $@)
	python generate_kernels.py

blocksparse/blocksparse_ops.so: $(OBJS) $(CU_OBJS)
	g++ $^ -shared -o $@ -L$(TF_LIB) -L$(NV_LIB) -ltensorflow_framework -lcudart -lcuda -L$(NCCL_LIB) -L$(MPI_LIB) -lnccl -lmpi

$(TARGET)/%.cu.o: src/%.cu $(TARGET)/blocksparse_kernels.h
	mkdir -p $(shell dirname $@)
	nvcc $(NVCCFLAGS) -c $< -o $@

$(TARGET)/%.o: src/%.cc src/*.h $(TARGET)/blocksparse_kernels.h
	mkdir -p $(shell dirname $@)
	g++ $(CCFLAGS) -c $< -o $@


# bazel-0.17.1-installer-linux-x86_64.sh (--user)
# NVIDIA-Linux-x86_64-396.37.run
# cuda_9.2.148_396.37_linux
# cudnn-9.2-linux-x64-v7.2.1.38.tgz
# nccl_2.2.13-1+cuda9.2_x86_64.txz

# apt-get install mpich

# uncomment:
# https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/core/kernels/batch_matmul_op_real.cc#L35


# ls -l /usr/local
# lrwxrwxrwx  1 root  root    19 Jul 14 13:11 cuda -> /usr/local/cuda-9.2/
# drwxr-xr-x 18 root  root  4096 Sep 14 16:12 cuda-9.2/
# lrwxrwxrwx  1 root  root    39 Jul 12 17:01 nccl -> /usr/local/nccl_2.2.13-1+cuda9.2_x86_64/
# drwxr-xr-x  4 root  root  4096 Jul 12 16:27 nccl_2.2.13-1+cuda9.2_x86_64/

# export TF_NEED_CUDA=1
# export TF_NEED_MKL=0
# export TF_NEED_GCP=0
# export TF_NEED_HDFS=0
# export TF_NEED_OPENCL=0
# export TF_NEED_AWS=0
# export TF_NEED_JEMALLOC=0
# export TF_NEED_KAFKA=0
# export TF_NEED_OPENCL_SYCL=0
# export TF_NEED_COMPUTECPP=0
# export TF_CUDA_CLANG=0
# export TF_NEED_TENSORRT=0
# export TF_ENABLE_XLA=0
# export TF_NEED_GDR=0
# export TF_NEED_VERBS=0
# export TF_NEED_MPI=0
# export TF_CUDA_VERSION="9.2"
# export TF_CUDNN_VERSION="7.2"
# export TF_NCCL_VERSION="2.2"
# export TF_CUDA_COMPUTE_CAPABILITIES="6.0,7.0"
# export GCC_HOST_COMPILER_PATH="/usr/bin/gcc"
# export CUDA_TOOLKIT_PATH="/usr/local/cuda"
# export CUDNN_INSTALL_PATH="/usr/local/cuda"
# export NCCL_INSTALL_PATH="/usr/local/nccl"

# alias tfbuild0="bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package"
# alias tfbuild1="bazel-bin/tensorflow/tools/pip_package/build_pip_package ~"
# alias tfbuild2="pip uninstall tensorflow"
# alias tfbuild3="pip install ~/tensorflow-*.whl"

# git clone blocksparse
# make compile
# pip install dist/*.whl