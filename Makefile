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
	-arch=sm_61 \
  	-gencode=arch=compute_35,code=sm_35 \
 	-gencode=arch=compute_50,code=sm_50 \
 	-gencode=arch=compute_52,code=sm_52 \
	-gencode=arch=compute_60,code=sm_60 \
	-gencode=arch=compute_61,code=sm_61 \
	-gencode=arch=compute_70,code=sm_70 \
	-gencode=arch=compute_61,code=compute_61

OBJS=\
	$(TARGET)/batch_norm_op.o \
	$(TARGET)/blocksparse_conv_op.o \
	$(TARGET)/blocksparse_kernels.o \
	$(TARGET)/blocksparse_l2_norm_op.o \
	$(TARGET)/blocksparse_matmul_op.o \
	$(TARGET)/cwise_linear_op.o \
	$(TARGET)/edge_bias_op.o \
	$(TARGET)/ew_op.o \
	$(TARGET)/gpu_types.o \
	$(TARGET)/layer_norm_op.o \
	$(TARGET)/nccl_op.o \
	$(TARGET)/lstm_op.o \
	$(TARGET)/optimize_op.o \
	$(TARGET)/quantize_op.o \
	$(TARGET)/transformer_op.o \
	$(TARGET)/embedding_op.o \
	$(TARGET)/matmul_op.o

CU_OBJS=\
	$(TARGET)/batch_norm_op_gpu.cu.o \
	$(TARGET)/blocksparse_l2_norm_op_gpu.cu.o \
	$(TARGET)/blocksparse_matmul_op_gpu.cu.o \
	$(TARGET)/blocksparse_matmul_gated_op_gpu.cu.o \
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
	g++ $^ -shared -o $@ -L$(TF_LIB) -L$(NV_LIB) -L$(NCCL_LIB) -L$(MPI_LIB) -ltensorflow_framework -lcudart -lcuda -lnccl -lmpi

$(TARGET)/%.cu.o: src/%.cu $(TARGET)/blocksparse_kernels.h
	mkdir -p $(shell dirname $@)
	nvcc $(NVCCFLAGS) -c $< -o $@

$(TARGET)/%.o: src/%.cc src/*.h $(TARGET)/blocksparse_kernels.h
	mkdir -p $(shell dirname $@)
	g++ $(CCFLAGS) -c $< -o $@


