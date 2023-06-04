.PHONY: all clean

CC=gcc -g -Wall

NVCC=nvcc
CUDAFLAGS= -std=c++11 -O2 -lineinfo

all: main.o general_functions.o dynamic_array.o libtest.so vector.o general_functions_par.o prefix_scan.o max_distance.o minmax.o
	$(NVCC) $(CUDAFLAGS) -Xcompiler -fPIC -o prog.out main.o general_functions.o dynamic_array.o test.o vector.o general_functions_par.o prefix_scan.o max_distance.o minmax.o -lm


main.o: main.cu Test/test.h Data_Types/data_types.h Sequential/general_functions.h Sequential/dynamic_array.h Data_Types/vector.h 
	$(NVCC) $(CUDAFLAGS) -Xcompiler -fPIC -c main.cu


libtest.so: test.o general_functions.o dynamic_array.o vector.o general_functions_par.o prefix_scan.o Data_Types/data_types.h Sequential/general_functions.h Sequential/dynamic_array.h Data_Types/vector.h Test/test.h Parallel/general_functions_par.h Parallel/prefix_scan.h
	$(NVCC) $(CUDAFLAGS) -Xcompiler -fPIC -shared -o libtest.so test.o general_functions.o dynamic_array.o vector.o general_functions_par.o prefix_scan.o


test.o: Test/test.cu Test/test.h Data_Types/data_types.h Sequential/general_functions.h Sequential/dynamic_array.h Data_Types/vector.h Parallel/general_functions_par.h Parallel/prefix_scan.h
	$(NVCC) $(CUDAFLAGS) -Xcompiler -fPIC -c Test/test.cu

general_functions.o: Sequential/general_functions.cu Data_Types/data_types.h Sequential/general_functions.h
	$(NVCC) $(CUDAFLAGS) -Xcompiler -fPIC -c Sequential/general_functions.cu


dynamic_array.o: Sequential/dynamic_array.cu Data_Types/data_types.h Sequential/dynamic_array.h
	$(NVCC) $(CUDAFLAGS) -Xcompiler -fPIC -c Sequential/dynamic_array.cu


vector.o: Data_Types/vector.cu Data_Types/data_types.h Data_Types/vector.h
	$(NVCC) $(CUDAFLAGS) -Xcompiler -fPIC -c Data_Types/vector.cu


general_functions_par.o: Parallel/general_functions_par.cu Parallel/general_functions_par.h Data_Types/data_types_par.h
	$(NVCC) $(CUDAFLAGS) -Xcompiler -fPIC -c Parallel/general_functions_par.cu


prefix_scan.o: Parallel/prefix_scan.cu Parallel/prefix_scan.h Data_Types/data_types_par.h Test/test.h
	$(NVCC) $(CUDAFLAGS) -Xcompiler -fPIC -c Parallel/prefix_scan.cu

max_distance.o: Parallel/max_distance.cu Parallel/max_distance.h Data_Types/data_types_par.h Test/test.h
	$(NVCC) $(CUDAFLAGS) -Xcompiler -fPIC -c Parallel/max_distance.cu

minmax.o: Parallel/minmax.cu Parallel/minmax.h Data_Types/data_types_par.h Test/test.h
	$(NVCC) $(CUDAFLAGS) -Xcompiler -fPIC -c Parallel/minmax.cu

clean:
	rm -rf prog *.o *.so
