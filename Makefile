NVCC_FLAGS := -O2 -std=c++14 --expt-relaxed-constexpr -w --generate-code arch=compute_70,code=sm_70

TARGET := getDoubletsFromHisto

all: $(TARGET)

clean:
	rm -f $(TARGET)

$(TARGET): $(TARGET).cu
	nvcc $(NVCC_FLAGS) -o $@ $(TARGET).cu
