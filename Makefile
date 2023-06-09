.PHONY: all clean

NVCC=nvcc
CUDAFLAGS= -std=c++11 -O2 -lineinfo

all: prog.out

prog.out: main.o general_functions.o dynamic_array.o vector.o general_functions_par.o prefix_scan.o max_distance.o minmax.o split.o data_types.o data_types_par.o test.o
	$(NVCC) $(CUDAFLAGS) -o prog.out main.o general_functions.o dynamic_array.o test.o vector.o general_functions_par.o prefix_scan.o max_distance.o minmax.o split.o data_types.o data_types_par.o -lm


main.o: main.cu Test/test.h
	$(NVCC) $(CUDAFLAGS) -c main.cu


test.o: Test/test.cu Test/test.h
	$(NVCC) $(CUDAFLAGS) -c Test/test.cu

data_types.o: Data_Types/data_types.cu Data_Types/data_types.h
	$(NVCC) $(CUDAFLAGS) -c Data_Types/data_types.cu

general_functions.o: Sequential/general_functions.cu Sequential/general_functions.h
	$(NVCC) $(CUDAFLAGS) -c Sequential/general_functions.cu


dynamic_array.o: Sequential/dynamic_array.cu Sequential/dynamic_array.h
	$(NVCC) $(CUDAFLAGS) -c Sequential/dynamic_array.cu


vector.o: Data_Types/vector.cu Data_Types/vector.h
	$(NVCC) $(CUDAFLAGS) -c Data_Types/vector.cu


data_types_par.o: Data_Types/data_types_par.cu Data_Types/data_types_par.h
	$(NVCC) $(CUDAFLAGS) -c Data_Types/data_types_par.cu

general_functions_par.o: Parallel/general_functions_par.cu Parallel/general_functions_par.h
	$(NVCC) $(CUDAFLAGS) -c Parallel/general_functions_par.cu


prefix_scan.o: Parallel/prefix_scan.cu Parallel/prefix_scan.h
	$(NVCC) $(CUDAFLAGS) -c Parallel/prefix_scan.cu

max_distance.o: Parallel/max_distance.cu Parallel/max_distance.h Data_Types/data_types_par.h Test/test.h
	$(NVCC) $(CUDAFLAGS) -c Parallel/max_distance.cu

minmax.o: Parallel/minmax.cu Parallel/minmax.h Data_Types/data_types_par.h Test/test.h
	$(NVCC) $(CUDAFLAGS) -c Parallel/minmax.cu

split.o: Parallel/split.cu Parallel/split.h
	$(NVCC) $(CUDAFLAGS) -c Parallel/split.cu

clean:
	rm -rf prog.out *.o *.so