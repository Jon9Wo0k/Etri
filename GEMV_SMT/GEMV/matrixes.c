#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <immintrin.h>

#include "port.h"
#include "proc_spec.h"
#include "matrixes.h"

FVector *alloc_vector(unsigned length)
   {
   FVector *rv = (FVector *)malloc(sizeof(FVector));
   if (!rv || !(rv->data = malloc(sizeof(vectortype_t) * length)))
      return NULL;
   rv->length = length;
   return rv;
   }

void delete_vector(FVector *v)
   {
   free(v->data);
   free(v);
   }

FMatrix *alloc_matrix(unsigned width,unsigned height)
   {
   FMatrix *rv = (FMatrix *)malloc(sizeof(FMatrix));
   if (!rv || !(rv->data = malloc(sizeof(matrixtype_t) * width * height)))
      return NULL;
   rv->width = width, rv->height = height;
   return rv;
   }

void delete_matrix(FMatrix *m)
   {
   free(m->data);
   free(m);
   }

void fill_by_zeroes(matrixtype_t *data,unsigned width,unsigned height)
   {
   memset(data,0,sizeof(matrixtype_t) * width * height);
   }

void fill_by_ones(matrixtype_t *data,unsigned width,unsigned height)
   {
   unsigned i,j;
   for (i = 0; i < height; i++)
      for (j = 0; j < width; j++)
         data[i * width + j] = 1.0;
   }

void fill_by_pattern(matrixtype_t *data,unsigned width,unsigned height)
   {
   unsigned i,j;
   for (i = 0; i < height; i++)
      {
      int val = 1;
      for (j = 0; j < width; j++)
         {
         data[i * width + j] = (float)val;
         val = val % 3 + 1;
         }
      }
   }

void fill_by_pattern_gemv(matrixtype_t *data,unsigned width,unsigned height)
   {
   unsigned i,j;
   for (i = 0; i < height; i++)
      {
      int val = 1;
      for (j = 0; j < width; j++)
         {
         data[i * width + j] = (float)val;
         val = val % 3 + 1;
         }
      }
   }

void fill_by_ones_gemv(vectortype_t *data,unsigned length)
   {
      unsigned i;
      for (i = 0; i < length; i++)
         data[i] = 1.0;
   }

void fill_by_zeroes_gemv(vectortype_t *data,unsigned length)
   {
   memset(data,0,sizeof(vectortype_t) * length);
   }

int check_result(matrixtype_t *data,unsigned width,unsigned height,unsigned k)
   {
   unsigned i,j;
   unsigned adds[3] = {0,1,3};

   float expected = (float)(6 * (k / 3) + adds[k % 3]);

   for (i = 0; i < height; i++)
      for (j = 0; j < width; j++)
         if ((unsigned)data[i * width + j] != expected)
            {
#ifdef _DEBUG
#ifdef _WIN32
            __debugbreak();
#else
            __asm__ volatile("int $0x03");
#endif
#endif
            return 1;
            }
   return 0;
   }


int check_result_gemv(vectortype_t *data, unsigned length, unsigned k)
{
   unsigned i;
   unsigned adds[3] = {0,1,3};

   // 예상값 계산 (2304 기준 4608.0)
   float expected = (float)(6 * (k / 3) + adds[k % 3]);

   for (i = 0; i < length; i++) {
      // [수정] 차이(diff)를 구해서 비교
      float diff = data[i] - expected;
      if (diff < 0) diff = -diff; // 절댓값 (fabs 대용)

      // 오차가 0.1보다 크면 진짜 틀린 것
      if (diff > 0.1f) 
      {
         // 디버깅용 출력 (첫 번째 에러만 출력하고 종료)
         // printf("Error: index %d, val %.5f != %.5f\n", i, data[i], expected); 
         return 1; 
      }
   }
   return 0; // 성공
}