#!/bin/bash

# BLOCKSIZES=(64 128 256 512 1024)
BLOCKSIZES=(256 512 1024)
MEMORY_MODELS=(1 2 3)

make clean
for b in "${BLOCKSIZES[@]}"; do
  for m in "${MEMORY_MODELS[@]}"; do
    make BLOCKSIZE=$b MEMORY_MODEL=$m
    ./prog.out
    make clean
  done
done
