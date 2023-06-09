.PHONY: all clean

CC=gcc -g -Wall



all: prog.out libtest.so 

prog.out: main.o test.o data_types.o general_functions.o dynamic_array.o vector.o general_functions.h test.h
	$(CC) -fopenmp -o prog.out main.o test.o general_functions.o dynamic_array.o vector.o data_types.o -lm


libtest.so: test.o data_types.o general_functions.o dynamic_array.o vector.o general_functions.h test.h 
	$(CC) -fPIC -shared -o libtest.so test.o general_functions.o dynamic_array.o vector.o data_types.o 


main.o: main.c test.h
	$(CC) -fopenmp -c main.c

test.o: test.c test.h
	$(CC) -fPIC -c test.c

data_types.o: data_types.c data_types.h
	$(CC) -fPIC -c data_types.c

general_functions.o: general_functions.c general_functions.h
	$(CC) -fopenmp -fPIC -c general_functions.c


dynamic_array.o: dynamic_array.c dynamic_array.h
	$(CC) -fPIC -c dynamic_array.c


vector.o: vector.c vector.h
	$(CC) -fPIC -c vector.c


clean:
	rm -rf prog.out *.o *.so