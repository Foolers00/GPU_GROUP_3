.PHONY: all clean

NVCC=nvcc
CUDAFLAGS= -std=c++14 -O2 -lineinfo

all: thrust_split.o thrust_general_functions.o

thrust_split.o: thrust_split.cu thrust_split.h
	$(NVCC) $(CUDAFLAGS) -c thrust_split.cu

thrust_general_functions.o: thrust_general_functions.cu thrust_general_functions.h
	$(NVCC) $(CUDAFLAGS) -c thrust_general_functions.cu

vector.o: ../Data_Types/vector.cu ../Data_Types/vector.h
	$(NVCC) $(CUDAFLAGS) -c ../Data_Types/vector.cu

clean:
	rm -rf prog.out *.o *.so