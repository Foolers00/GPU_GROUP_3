.PHONY: all clean

CC=gcc -g -Wall

all: main.o general_functions.o dynamic_array.o libtest.so
	$(CC) -o prog main.o general_functions.o dynamic_array.o test.o -lm 


main.o: main.c test.h data_types.h general_functions.h dynamic_array.h 
	$(CC) -c main.c


libtest.so: test.o general_functions.o dynamic_array.o data_types.h general_functions.h dynamic_array.h
	$(CC) -fPIC -shared -o libtest.so test.o general_functions.o dynamic_array.o


test.o: test.c test.h data_types.h general_functions.h dynamic_array.h
	$(CC) -c -fPIC test.c

general_functions.o: general_functions.c data_types.h general_functions.h
	$(CC) -c -fPIC general_functions.c


dynamic_array.o: dynamic_array.c data_types.h dynamic_array.h
	$(CC) -c -fPIC dynamic_array.c


clean:
	rm -rf prog main.o general_functions.o dynamic_array.o test.o libtest.so
