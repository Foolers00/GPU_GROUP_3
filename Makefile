all: main.o general_functions.o dynamic_array.o
	gcc -o prog main.o general_functions.o dynamic_array.o


main.o: main.c data_types.h general_functions.h dynamic_array.h 
	gcc -c main.c


general_functions.o: general_functions.c data_types.h general_functions.h
	gcc -c general_functions.c


dynamic_array.o: dynamic_array.c data_types.h dynamic_array.h
	gcc -c dynamic_array.c


clean:
	rm -rf prog main.o general_functions.o dynamic_array.o
