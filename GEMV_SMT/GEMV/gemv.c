#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>
#include <assert.h>
#include <pthread.h>
#include <unistd.h>
#include <sched.h>

// [필수 헤더]
#include "port.h"
#include "proc_spec.h"
#include "matrixes.h"
#include "calc_env.h"

// =================================================================================
// [설정] 파라미터 및 매크로
// =================================================================================
#define ROWS_PER_THREAD 2 
#define MAIN_CORE 0
#define SIBLING_CORE 1

#ifdef _DEBUG
#define CYCLE_COUNT 20
#else
#define CYCLE_COUNT 50
#endif

// 동기화 및 양자화 설정
#define CMD_EXIT 0xFFFFFFFFFFFFFFFF
#define CMD_IDLE 0
#define PACK_CMD(k, m) (((uint64_t)(m) << 32) | (uint64_t)(k))

// 양자화 스케일 (간단한 데모를 위해 고정 스케일 사용)
// 실제로는 Block 단위 max값을 계산해야 하지만, 여기서는 메모리 대역폭 효과 확인용
#define QUANT_SCALE 127.0f 
#define DEQUANT_SCALE (1.0f / 127.0f)

// =================================================================================
// [Utils] Helper Functions
// =================================================================================

static inline float hsum_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static int64_t get_nanotime_local() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}


// =================================================================================
// [Benchmark] Baseline & Measure
// =================================================================================

// Baseline (Reference) - 변경 없음 (순수 Float)
void kernel_no_pack_1xK(int k_size, const float *A, int lda, const float *B, float *C) {
    __m256 sum0 = _mm256_setzero_ps(); 
    __m256 sum1 = _mm256_setzero_ps();

    const float* A0 = A + 0 * lda; 

    for (int k = 0; k < k_size; k += 8) {
        __m256 b_vec = _mm256_loadu_ps(&B[k]);
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(A0+k), b_vec, sum0);
    }
    C[0] += hsum_avx2(sum0); 
}

void kernel_no_pack_2xK(int k_size, const float *A, int lda, const float *B, float *C) {
    __m256 sum0 = _mm256_setzero_ps(); 
    __m256 sum1 = _mm256_setzero_ps();

    const float* A0 = A + 0 * lda; 
    const float* A1 = A + 1 * lda;

    for (int k = 0; k < k_size; k += 8) {
        __m256 b_vec = _mm256_loadu_ps(&B[k]);
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(A0+k), b_vec, sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(A1+k), b_vec, sum1);
    }
    C[0] += hsum_avx2(sum0); 
    C[1] += hsum_avx2(sum1);
}

// 2. [4 Rows] 적절한 균형
void kernel_no_pack_4xK(int k_size, const float *A, int lda, const float *B, float *C) {
    __m256 sum0 = _mm256_setzero_ps(); __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps(); __m256 sum3 = _mm256_setzero_ps();

    const float* A0 = A + 0 * lda; const float* A1 = A + 1 * lda;
    const float* A2 = A + 2 * lda; const float* A3 = A + 3 * lda;

    for (int k = 0; k < k_size; k += 8) {
        __m256 b_vec = _mm256_loadu_ps(&B[k]);
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(A0+k), b_vec, sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(A1+k), b_vec, sum1);
        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(A2+k), b_vec, sum2);
        sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(A3+k), b_vec, sum3);
    }
    C[0] += hsum_avx2(sum0); C[1] += hsum_avx2(sum1);
    C[2] += hsum_avx2(sum2); C[3] += hsum_avx2(sum3);
}

// 3. [8 Rows] 레지스터 활용 극대화 (기존 코드)
void kernel_no_pack_8xK(int k_size, const float *A, int lda, const float *B, float *C) {
    __m256 sum0 = _mm256_setzero_ps(); __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps(); __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps(); __m256 sum5 = _mm256_setzero_ps();
    __m256 sum6 = _mm256_setzero_ps(); __m256 sum7 = _mm256_setzero_ps();

    const float* A0 = A + 0 * lda; const float* A1 = A + 1 * lda;
    const float* A2 = A + 2 * lda; const float* A3 = A + 3 * lda;
    const float* A4 = A + 4 * lda; const float* A5 = A + 5 * lda;
    const float* A6 = A + 6 * lda; const float* A7 = A + 7 * lda;

    for (int k = 0; k < k_size; k += 8) {
        __m256 b_vec = _mm256_loadu_ps(&B[k]);
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(A0+k), b_vec, sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(A1+k), b_vec, sum1);
        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(A2+k), b_vec, sum2);
        sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(A3+k), b_vec, sum3);
        sum4 = _mm256_fmadd_ps(_mm256_loadu_ps(A4+k), b_vec, sum4);
        sum5 = _mm256_fmadd_ps(_mm256_loadu_ps(A5+k), b_vec, sum5);
        sum6 = _mm256_fmadd_ps(_mm256_loadu_ps(A6+k), b_vec, sum6);
        sum7 = _mm256_fmadd_ps(_mm256_loadu_ps(A7+k), b_vec, sum7);
    }
    C[0] += hsum_avx2(sum0); C[1] += hsum_avx2(sum1);
    C[2] += hsum_avx2(sum2); C[3] += hsum_avx2(sum3);
    C[4] += hsum_avx2(sum4); C[5] += hsum_avx2(sum5);
    C[6] += hsum_avx2(sum6); C[7] += hsum_avx2(sum7);
}

THREAD_FUNC ThreadProc_SMT_1x1(void *lpParameter) {
    VCalcParams *params = (VCalcParams *)lpParameter;
    volatile uint64_t *sync_cmd = &params->sinhronizer;
    volatile uint64_t *sync_ack = &params->sinhronizer2;

    if (get_thread_processor() != SIBLING_CORE) params->error = 1;

    while (1) {
        // 1. Wait
        uint64_t cmd = *sync_cmd;
        while (cmd == CMD_IDLE) { _mm_pause(); cmd = *sync_cmd; }
        if (cmd == CMD_EXIT) break;

        // 2. Range (Bottom Half)
        int half_point = params->m_size / 2;
        // 2행 단위 정렬 (2x 커널이므로 2의 배수여야 함)
        if (half_point % 2 != 0) half_point = (half_point / 2) * 2;

        int m_start = half_point;
        int m_end = params->m_size;
        int lda = params->k_size;

        // 3. Compute (2x Unroll)
        for (int m = m_start; m < m_end; m += 1) {
            if (m + 1 <= m_end) {
                kernel_no_pack_1xK(params->k_size, &params->A[(size_t)m * lda], lda, params->B, &params->C[m]);
            }
        }

        // 4. Ack & Handshake
        *sync_ack = cmd;
        while (*sync_cmd != CMD_IDLE && *sync_cmd != CMD_EXIT) { _mm_pause(); }
    }
    THREAD_RETURN(0);
}

// [Main Driver] 상반부 (2x 커널 사용)
void run_smt_2_split(VCalcParams *params) {
    params->sinhronizer = CMD_IDLE;
    params->sinhronizer2 = CMD_IDLE;

    // 1. Wake up
    params->sinhronizer = 1; 

    // 2. Range (Top Half)
    int half_point = params->m_size / 2;
    if (half_point % 2 != 0) half_point = (half_point / 2) * 2;

    int m_start = 0;
    int m_end = half_point;
    int lda = params->k_size;

    // 3. Compute (Main Work)
    for (int m = m_start; m < m_end; m += 1) {
        if (m + 1 <= m_end) {
            kernel_no_pack_1xK(params->k_size, &params->A[(size_t)m * lda], lda, params->B, &params->C[m]);
        }
    }

    // 4. Wait
    while (params->sinhronizer2 != 1) { _mm_pause(); }

    // 5. Reset
    params->sinhronizer2 = CMD_IDLE;
    params->sinhronizer = CMD_IDLE;
    params->sinhronizer = CMD_EXIT; 
}

// [Sibling Thread] 하반부 (2x 커널 사용)
THREAD_FUNC ThreadProc_SMT_2x2(void *lpParameter) {
    VCalcParams *params = (VCalcParams *)lpParameter;
    volatile uint64_t *sync_cmd = &params->sinhronizer;
    volatile uint64_t *sync_ack = &params->sinhronizer2;

    if (get_thread_processor() != SIBLING_CORE) params->error = 1;

    while (1) {
        // 1. Wait
        uint64_t cmd = *sync_cmd;
        while (cmd == CMD_IDLE) { _mm_pause(); cmd = *sync_cmd; }
        if (cmd == CMD_EXIT) break;

        // 2. Range (Bottom Half)
        int half_point = params->m_size / 2;
        // 2행 단위 정렬 (2x 커널이므로 2의 배수여야 함)
        if (half_point % 2 != 0) half_point = (half_point / 2) * 2;

        int m_start = half_point;
        int m_end = params->m_size;
        int lda = params->k_size;

        // 3. Compute (2x Unroll)
        for (int m = m_start; m < m_end; m += 2) {
            if (m + 2 <= m_end) {
                kernel_no_pack_2xK(params->k_size, &params->A[(size_t)m * lda], lda, params->B, &params->C[m]);
            }
        }

        // 4. Ack & Handshake
        *sync_ack = cmd;
        while (*sync_cmd != CMD_IDLE && *sync_cmd != CMD_EXIT) { _mm_pause(); }
    }
    THREAD_RETURN(0);
}

// [Main Driver] 상반부 (2x 커널 사용)
void run_smt_4_split(VCalcParams *params) {
    params->sinhronizer = CMD_IDLE;
    params->sinhronizer2 = CMD_IDLE;

    // 1. Wake up
    params->sinhronizer = 1; 

    // 2. Range (Top Half)
    int half_point = params->m_size / 2;
    if (half_point % 2 != 0) half_point = (half_point / 2) * 2;

    int m_start = 0;
    int m_end = half_point;
    int lda = params->k_size;

    // 3. Compute (Main Work)
    for (int m = m_start; m < m_end; m += 2) {
        if (m + 2 <= m_end) {
            kernel_no_pack_2xK(params->k_size, &params->A[(size_t)m * lda], lda, params->B, &params->C[m]);
        }
    }

    // 4. Wait
    while (params->sinhronizer2 != 1) { _mm_pause(); }

    // 5. Reset
    params->sinhronizer2 = CMD_IDLE;
    params->sinhronizer = CMD_IDLE;
    params->sinhronizer = CMD_EXIT; 
}

// =================================================================================
// [Drivers] 각 Unrolling 전략별 실행 함수
// =================================================================================

// [아래 코드를 기존 코드의 run_16x1 함수 밑에 추가하세요]

// =================================================================================
// [Strategy 5] SMT Split (16x -> 8x + 8x)
// 전체 M을 절반으로 나누어, Main과 Sibling이 각각 8xK 커널로 동시에 처리
// =================================================================================

// [Sibling Thread] 하반부 (4x 커널 사용)
THREAD_FUNC ThreadProc_SMT_4x4(void *lpParameter) {
    VCalcParams *params = (VCalcParams *)lpParameter;
    volatile uint64_t *sync_cmd = &params->sinhronizer;
    volatile uint64_t *sync_ack = &params->sinhronizer2;

    if (get_thread_processor() != SIBLING_CORE) params->error = 1;

    while (1) {
        // 1. Wait
        uint64_t cmd = *sync_cmd;
        while (cmd == CMD_IDLE) { _mm_pause(); cmd = *sync_cmd; }
        if (cmd == CMD_EXIT) break;

        // 2. Range (Bottom Half)
        int half_point = params->m_size / 2;
        // 4행 단위 정렬 (4x 커널이므로 4의 배수여야 함)
        if (half_point % 4 != 0) half_point = (half_point / 4) * 4;

        int m_start = half_point;
        int m_end = params->m_size;
        int lda = params->k_size;

        // 3. Compute (4x Unroll)
        for (int m = m_start; m < m_end; m += 4) {
            if (m + 4 <= m_end) {
                kernel_no_pack_4xK(params->k_size, &params->A[(size_t)m * lda], lda, params->B, &params->C[m]);
            }
        }

        // 4. Ack & Handshake
        *sync_ack = cmd;
        while (*sync_cmd != CMD_IDLE && *sync_cmd != CMD_EXIT) { _mm_pause(); }
    }
    THREAD_RETURN(0);
}

// [Main Driver] 상반부 (4x 커널 사용)
void run_smt_8_split(VCalcParams *params) {
    params->sinhronizer = CMD_IDLE;
    params->sinhronizer2 = CMD_IDLE;

    // 1. Wake up
    params->sinhronizer = 1; 

    // 2. Range (Top Half)
    int half_point = params->m_size / 2;
    if (half_point % 4 != 0) half_point = (half_point / 4) * 4;

    int m_start = 0;
    int m_end = half_point;
    int lda = params->k_size;

    // 3. Compute (Main Work)
    for (int m = m_start; m < m_end; m += 4) {
        if (m + 4 <= m_end) {
            kernel_no_pack_4xK(params->k_size, &params->A[(size_t)m * lda], lda, params->B, &params->C[m]);
        }
    }

    // 4. Wait
    while (params->sinhronizer2 != 1) { _mm_pause(); }

    // 5. Reset
    params->sinhronizer2 = CMD_IDLE;
    params->sinhronizer = CMD_IDLE;
    params->sinhronizer = CMD_EXIT; 
}

// [Sibling Thread] 하반부(Bottom Half) 처리
THREAD_FUNC ThreadProc_SMT_8x8(void *lpParameter) {
    VCalcParams *params = (VCalcParams *)lpParameter;
    volatile uint64_t *sync_cmd = &params->sinhronizer;  // Main -> Sibling
    volatile uint64_t *sync_ack = &params->sinhronizer2; // Sibling -> Main

    if (get_thread_processor() != SIBLING_CORE) params->error = 1;

    while (1) {
        // 1. Wait for Command
        uint64_t cmd = *sync_cmd;
        while (cmd == CMD_IDLE) {
            _mm_pause();
            cmd = *sync_cmd;
        }

        if (cmd == CMD_EXIT) break;

        // 2. Calculate Range (Bottom Half)
        // 전체 행렬을 반으로 나누어 뒷부분을 맡음
        int half_point = params->m_size / 2;
        // 8단위 정렬 (안전을 위해)
        if (half_point % 8 != 0) half_point = (half_point / 8) * 8;

        int m_start = half_point;
        int m_end = params->m_size;
        int lda = params->k_size;

        // 3. Compute Loop (8x Unroll)
        for (int m = m_start; m < m_end; m += 8) {
            if (m + 8 <= m_end) {
                kernel_no_pack_8xK(params->k_size, &params->A[(size_t)m * lda], lda, params->B, &params->C[m]);
            }
        }

        // 4. Signal Done (Ack)
        *sync_ack = cmd;

        // 5. Wait for Reset (Handshake)
        while (*sync_cmd != CMD_IDLE && *sync_cmd != CMD_EXIT) {
            _mm_pause();
        }
    }
    THREAD_RETURN(0);
}

// [Main Driver] 상반부(Top Half) 처리 + Sync
void run_smt_16_split(VCalcParams *params) {
    params->sinhronizer = CMD_IDLE;
    params->sinhronizer2 = CMD_IDLE;

    // 1. Wake up Sibling (Start)
    // 전체 통으로 처리하므로 CMD에는 1(RUN)만 보내면 됨
    params->sinhronizer = 1; 

    // 2. Calculate Range (Top Half)
    int half_point = params->m_size / 2;
    if (half_point % 8 != 0) half_point = (half_point / 8) * 8;

    int m_start = 0;
    int m_end = half_point;
    int lda = params->k_size;

    // 3. Compute Loop (Main Thread Work)
    for (int m = m_start; m < m_end; m += 8) {
        if (m + 8 <= m_end) {
            kernel_no_pack_8xK(params->k_size, &params->A[(size_t)m * lda], lda, params->B, &params->C[m]);
        }
    }

    // 4. Wait for Sibling
    while (params->sinhronizer2 != 1) {
        _mm_pause();
    }

    // 5. Reset Flags
    params->sinhronizer2 = CMD_IDLE;
    params->sinhronizer = CMD_IDLE;
    
    // 종료 신호(CMD_EXIT)는 measure 함수가 스레드를 정리할 때 보냄
    // (여기서는 단일 실행만 보장하면 됨)
    params->sinhronizer = CMD_EXIT; 
}


// Strategy 1: Unroll 2
void run_2x1(VCalcParams *params) {
    int lda = params->k_size; 
    for (int m = 0; m < params->m_size; m += 2) {
        // 남은 행 처리 (Boundary Check)
        if (m + 2 <= params->m_size) {
            kernel_no_pack_2xK(params->k_size, &params->A[(size_t)m * lda], lda, params->B, &params->C[m]);
        } else {
            // Remainder handling (1 row left) - 생략 혹은 1xK 호출
        }
    }
}

// Strategy 2: Unroll 4
void run_4x1(VCalcParams *params) {
    int lda = params->k_size; 
    for (int m = 0; m < params->m_size; m += 4) {
        if (m + 4 <= params->m_size) {
            kernel_no_pack_4xK(params->k_size, &params->A[(size_t)m * lda], lda, params->B, &params->C[m]);
        }
    }
}

// Strategy 3: Unroll 8
void run_8x1(VCalcParams *params) {
    int lda = params->k_size; 
    for (int m = 0; m < params->m_size; m += 8) {
        if (m + 8 <= params->m_size) {
            kernel_no_pack_8xK(params->k_size, &params->A[(size_t)m * lda], lda, params->B, &params->C[m]);
        }
    }
}

// Strategy 4: Unroll 16 (8xK 커널을 2번 호출)
void run_16x1(VCalcParams *params) {
    int lda = params->k_size; 
    for (int m = 0; m < params->m_size; m += 16) {
        if (m + 8 <= params->m_size) {
            kernel_no_pack_8xK(params->k_size, &params->A[(size_t)m * lda], lda, params->B, &params->C[m]);
        }
        if (m + 16 <= params->m_size) {
            kernel_no_pack_8xK(params->k_size, &params->A[(size_t)(m + 8) * lda], lda, params->B, &params->C[m + 8]);
        }
    }
}

typedef void (*CMVFunc)(VCalcParams *params);

void measure(char *name, CMVFunc func, CThreadRoutine thread, VCalcParams *params) {
    int i, err = 0;
    double rv = -1.0;
    THREAD_ID_TYPE thid = 0;
    double avg = 0;

    printf("Measuring %s on core %d\n", name, get_thread_processor());
    if (get_thread_processor() != MAIN_CORE) {
        printf("%15s failed to run on cpu 2\n", name);
        return;
    }

    for (i = 0; i < CYCLE_COUNT; i++) {
        reset_params_gemv(params);
        
        if (thread) {
            thid = start_thread(thread, params, SIBLING_CORE);
        }

        int64_t t1 = get_nanotime_local();
        func(params); 
        int64_t t2 = get_nanotime_local();
        
        if (thread) {
            wait_thread(thid); 
        }
        
        if (params->error) { printf("%15s failed error\n", name); return; }
        
        double ops = 2.0 * (double)params->m_size * (double)params->k_size;
        double res = ops / ((double)(t2 - t1) / 1e9) / 1e9;
        
        avg += res;
        if (res > rv) rv = res;
        if (i == 0) 
            err += check_result_gemv(params->C, params->m_size, params->k_size);
    }
    avg /= (double)CYCLE_COUNT;
    printf("%15s %7.3f GFlops (Max: %7.3f) | Errors: %d\n", name, avg, rv, err);
}

int main(void){
    maximize_priority();
    set_thread_processor(MAIN_CORE);

    printf("=== GEMV Unrolling Factor Benchmark (2 vs 4 vs 8 vs 16) ===\n");

    for(int i = 1 ; i <= 3; i++){ // 테스트 횟수 단축
        int cur_M = 4096 * i; // 사이즈 좀 키움
        int cur_K = 4096 * i;

        FMatrix *A = alloc_matrix(cur_K, cur_M);
        FVector *B = alloc_vector(cur_K);
        FVector *C = alloc_vector(cur_M);
        
        VCalcParams *params = create_params_gemv(A, B, C, 0.75, 16);

        printf("\n[M:%d, K:%d]\n", cur_M, cur_K);

        // 1. Unroll 2
        // measure("Unroll_2x", run_2x1, NULL, params);
        // measure("SMT_Split_2 -> 1+1", run_smt_2_split, ThreadProc_SMT_1x1, params);
        // printf("\n");
        // 2. Unroll 4
        // measure("Unroll_4x", run_4x1, NULL, params);
        //  measure("SMT_Split_4->2+2", run_smt_4_split, ThreadProc_SMT_2x2, params);
        //  printf("\n");

        // 3. Unroll 8 (Sweet Spot 예상)
        // measure("Unroll_8x", run_8x1, NULL, params);
         measure("SMT_Split_8->4+4", run_smt_8_split, ThreadProc_SMT_4x4, params);
        // printf("\n");

        // 4. Unroll 16 (Instruction Cache 이득?)
        // measure("Unroll_16x", run_16x1, NULL, params);
        //  measure("SMT_Split_16->8+8", run_smt_16_split, ThreadProc_SMT_8x8, params); // 기존
        //  printf("\n");

        // Main 함수 안의 measure 호출부에 추가

        free_params_gemv(params);
        delete_matrix(A); delete_vector(B); delete_vector(C);
    }
    return 0;
}