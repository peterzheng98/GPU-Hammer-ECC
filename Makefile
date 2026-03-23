NVCC       ?= nvcc
CUDA_ARCH  ?= 80 90
NVCCFLAGS   = -O2 -std=c++14 -Xcompiler -Wall
NVCCFLAGS  += $(foreach a,$(CUDA_ARCH),-gencode arch=compute_$(a),code=sm_$(a))
LDFLAGS     = -lnvidia-ml -lpthread

TARGET = gpu_hammer
SRC    = src/gpu_hammer.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	@which $(NVCC) > /dev/null 2>&1 || { echo "Error: nvcc not found. Install CUDA Toolkit."; exit 1; }
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)
