#include <stdlib.h>

#include "port.h"
#include "calc_env.h"

#define COVER_COUNT(VALUE,ALIGN) ((VALUE) / (ALIGN) + ((VALUE) % (ALIGN) ? 1 : 0))

#ifdef L1_RESERVE
   #define L1_RESERVE_LINES (COVER_COUNT(sizeof(FCalcParams),CACHE_LINE_SIZE * 2) * 2)
#else
   #define L1_RESERVE_LINES 0
#endif

static unsigned _get_k_step(unsigned k_size,double L1_kt, unsigned core_width)
   {
   unsigned total_L1_cache_lines = L1_CACHE / CACHE_LINE_SIZE;
   unsigned L1_cache_lines = (unsigned)((double) total_L1_cache_lines * L1_kt); 
   if (total_L1_cache_lines < L1_cache_lines + L1_RESERVE_LINES)
      L1_cache_lines = total_L1_cache_lines - L1_RESERVE_LINES;

   return min(L1_cache_lines / (core_width / REGISTRIES_PER_CACHE_LINE), k_size) / 8 * 8; 
   }

static unsigned _get_k_step_gemv(unsigned k_size, double L2_kt)
{
    // 전체 L2 캐시 크기
    const unsigned total_L2_bytes = L2_CACHE;

    // 사용할 비율만큼 제한 (ex: 0.75)
    unsigned usable_L2_bytes = (unsigned)((double)total_L2_bytes * L2_kt);

    // 벡터 원소 기준 최대 개수
    unsigned max_k_elems = usable_L2_bytes / sizeof(vectortype_t);

    // 캐시에 다 들어오면 그대로 사용
    if (k_size <= max_k_elems) {
        return k_size;
    }

    // AVX 정렬 (8 단위 맞춤)
    return (max_k_elems / 8) * 8;
}


FCalcParams *create_params(const FMatrix *m1, const FMatrix *m2, FMatrix *res,double L1_kt, unsigned core_width, unsigned core_height)
   {
   FCalcParams *rv;

   matrixtype_t *micro_A,*micro_B;

   if (posix_memalign((void **)&micro_A,CACHE_LINE_SIZE * 2,L2_CACHE * 2))
      return NULL;

   if (posix_memalign((void **)&micro_B,CACHE_LINE_SIZE * 2,L3_CACHE))
      return NULL;

#ifdef L1_RESERVE
   rv = (FCalcParams *)micro_B;
#else
   if (posix_memalign((void **)&rv,CACHE_LINE_SIZE * 2,sizeof(FCalcParams)))
      return NULL;
#endif   
   rv->A = m1->data;
   rv->B = m2->data;
   rv->C = res->data;
   rv->micro_A = (matrixtype_t *)micro_A;
   rv->micro_B = (matrixtype_t *)(micro_B + L1_RESERVE_LINES * CACHE_LINE_SIZE);
   rv->n_size = res->width;
   rv->m_size = res->height;
   rv->k_size = m2->height;
   update_params(rv,L1_kt,core_width,core_height);

   return rv;
   }

void update_params(FCalcParams *params,double L1_kt, unsigned core_width, unsigned core_height)
   {
   unsigned kstep = params->k_step = _get_k_step(params->k_size,L1_kt,core_width);
   params->k_stride = kstep + L1_RESERVE_LINES;
   
   int ap_size = core_height * kstep;
   unsigned ap_step = ap_size * sizeof(matrixtype_t) / L1_STRIDE;
   if (1 || !ap_step)
      {
      params->ap_stride = core_height * kstep;
      params->m_step = min(L2_CACHE / (sizeof(matrixtype_t) * kstep), params->m_size) / (core_height * 2) * (core_height * 2); // height of micro_A to fit in L2 cache
      params->n_step = min(L3_CACHE / (sizeof(matrixtype_t) * kstep), params->n_size) / (core_width * ITEMS_PER_CACHE_LINE) * (core_width * ITEMS_PER_CACHE_LINE);
      return;
      }
   if (ap_size % L1_STRIDE)
      ap_step++;
   params->ap_stride = ap_step * L1_STRIDE / sizeof(matrixtype_t);

   int ap_cnt = L2_CACHE / sizeof(matrixtype_t) / params->ap_stride;
   params->m_step = min(ap_cnt * core_height, params->m_size) / core_height * core_height; // height of micro_A to fit in L2 cache
   params->n_step = min(L3_CACHE / (sizeof(matrixtype_t) * kstep), params->n_size) / (core_width * ITEMS_PER_CACHE_LINE) * (core_width * ITEMS_PER_CACHE_LINE);
   }

VCalcParams *create_params_gemv(const FMatrix *m1, const FVector *m2, FVector *res,double L1_kt, unsigned core_length)
   {
    VCalcParams *rv;

    matrixtype_t *micro_A;
    vectortype_t *micro_B;

   if (posix_memalign((void **)&micro_A,CACHE_LINE_SIZE * 2,L2_CACHE * 2))
      return NULL;

   if (posix_memalign((void **)&micro_B,CACHE_LINE_SIZE * 2,L3_CACHE))
      return NULL;

#ifdef L1_RESERVE
   rv = (FCalcParams *)micro_B;
#else
   if (posix_memalign((void **)&rv,CACHE_LINE_SIZE * 2,sizeof(VCalcParams)))
      return NULL;
#endif   
   rv->A = m1->data;
   rv->B = m2->data;
   rv->C = res->data;
   rv->micro_A = (matrixtype_t *)micro_A;
   rv->micro_B = NULL;
   rv->n_size = 1;
   rv->m_size = res->length;
   rv->k_size = m2->length;
   update_params_gemv(rv,L1_kt,core_length);

   return rv;
   }

void update_params_gemv(VCalcParams *params, double L2_kt, unsigned core_length)
{
    // K_step 계산 (L2 75% 제한)
    params->k_step = _get_k_step_gemv(params->k_size, L2_kt);

    params->k_stride = params->k_step; 
   
    // M blocking (동기화 단위)
    params->m_step = 512;

    params->n_step = 1;   
    
    params->ap_stride = params->k_size; 
}

   
void reset_params(FCalcParams *params)
   {
   fill_by_pattern(params->A,params->k_size,params->m_size);
   fill_by_ones(params->B,params->n_size,params->k_size);
   fill_by_zeroes(params->C,params->n_size,params->m_size);
   params->sinhronizer = SYNC_NOT_STARTED;
   params->sinhronizer2 = SYNC_NOT_STARTED;
   params->half_done[0] = params->half_done[1] = 0;
   params->alpha = 1.0;
   params->error = 0;
   }

void reset_params_gemv(VCalcParams *params)
   {
   fill_by_pattern_gemv(params->A,params->k_size,params->m_size);
   fill_by_ones_gemv(params->B,params->k_size);
   fill_by_zeroes_gemv(params->C,params->m_size);
   params->sinhronizer = SYNC_NOT_STARTED;
   params->sinhronizer2 = SYNC_NOT_STARTED;
   params->half_done[0] = params->half_done[1] = 0;
   params->alpha = 1.0;
   params->error = 0;
   }


void free_params(FCalcParams *params)
   {
#ifndef L1_RESERVE
   posix_memalign_free(params->micro_B);
#endif
   posix_memalign_free(params->micro_A);
   posix_memalign_free(params);
   }

void free_params_gemv(VCalcParams *params)
   {
#ifndef L1_RESERVE
#endif
   posix_memalign_free(params->micro_A);
   posix_memalign_free(params);
   }  