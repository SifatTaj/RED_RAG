.PHONY: cos_sim hash_bucket clean

CUDA_SRC_DIR = core/cuda
CUDA_OUT_DIR = core/cuda/lib

all: cos_sim hash_bucket

cos_sim:
	nvcc -shared -o  $(CUDA_OUT_DIR)/cos_sim.so --compiler-options '-fPIC' $(CUDA_SRC_DIR)/cos_sim.cu -lcublas

hash_bucket:
	nvcc -shared -o  $(CUDA_OUT_DIR)/hash_bucket.so --compiler-options '-fPIC' $(CUDA_SRC_DIR)/hash_bucket.cu -lcublas

clean:
	rm -r $(CUDA_OUT_DIR)/cos_sim.so $(CUDA_OUT_DIR)/hash_bucket.so