# 여기에 -mllvm -stats 추가
COMMON_OPTIONS = -O3 -w -Wall -std=c11 -mtune=native -march=native -mfma -mavx2  -D_GNU_SOURCE
LIBS = 

all:
	gcc $(COMMON_OPTIONS) `ls *.c | grep -v -E "memo|gemm.c"` -o matrix.out $(LIBS)

asm:
	clang $(COMMON_OPTIONS) -S `ls *.c | grep -v gemm.c`
