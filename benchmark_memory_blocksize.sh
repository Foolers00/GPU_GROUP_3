#!/bin/bash

# Define the folder path
folder_path="Data_Types"

# Define the file name
file_name="data_types_par.h"

# Define the options for BLOCKSIZE and MEMORY_MODEL
block_sizes=("32" "128" "512" "1024")
memory_models=("STD_MEMORY" "PINNED_MEMORY" "ZERO_MEMORY")

# Loop through each block size
for block_size in "${block_sizes[@]}"
do
    # Define the new value for line 18 based on the block size
    new_line_18="#define BLOCKSIZE $block_size"

    # Rewrite line 18
    sed -i "18s/.*/$new_line_18/" "$folder_path/$file_name"

    # Loop through each memory model
    for memory_model in "${memory_models[@]}"
    do
        # Define the new value for line 29 based on the memory model
        new_line_29="#define MEMORY_MODEL $memory_model"

        # Rewrite line 29
        sed -i "29s/.*/$new_line_29/" "$folder_path/$file_name"

        echo "Lines 18 and 29 of $folder_path/$file_name have been rewritten with the values:"
        echo "    - Line 18: BLOCKSIZE = $block_size"
        echo "    - Line 29: MEMORY_MODEL = $memory_model"
        echo "------------------------------"

        # Run make
        make

        # Execute prog.out
        ./prog.out

        make clean
    done
done
