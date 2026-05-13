#ifndef _MATRIXES_H
#define _MATRIXES_H

typedef float matrixtype_t;
typedef float vectortype_t;
typedef float scalartype_t;

typedef struct FVectorTg
   {
   unsigned length;
   matrixtype_t *data;
   } FVector;

typedef struct FMatrixTg
   {
   unsigned width;
   unsigned height;
   matrixtype_t *data;
   } FMatrix;

FMatrix *alloc_matrix(unsigned width,unsigned height);
FVector *alloc_vector(unsigned length);

void delete_vector(FVector *v);
void fill_by_zeroes_gemv(vectortype_t *data,unsigned length);
void fill_by_ones_gemv(vectortype_t *data,unsigned length);
void fill_by_pattern_gemv(matrixtype_t *data,unsigned width,unsigned height);
int check_result_gemv(vectortype_t *data,unsigned length,unsigned k);

void delete_matrix(FMatrix *m);
void fill_by_zeroes(matrixtype_t *data,unsigned width,unsigned height);
void fill_by_ones(matrixtype_t *data,unsigned width,unsigned height);
void fill_by_pattern(matrixtype_t *data,unsigned width,unsigned height);
int check_result(matrixtype_t *data,unsigned width,unsigned height,unsigned k);

#endif