# Etri

# SMT-GEMM / SMT-GEMV

physical core와 그 SMT(Simultaneous Multi-Threading, 하이퍼스레딩) sibling core를 함께 활용해 single-core 대비 throughput을 끌어올리는 AVX2/FMA 기반 SGEMM·SGEMV kernel 실험 프로젝트입니다. lock-free spin synchronization으로 두 thread가 작업을 절반씩 나눠 처리하고, 그 결과를 실측 GFLOPS로 비교합니다.

## 구성

| 디렉터리 | 내용 |
|---|---|
| `GEMM_SMT/` | matrix-matrix multiplication (SGEMM). 6×16, 2×48 microkernel + cache blocking |
| `GEMV_SMT/` | matrix-vector multiplication (SGEMV). unrolling factor(2/4/8/16)별 성능 비교 + SMT-split benchmark |

## Method

- **SMT core split**: main core와 sibling core가 matrix를 절반씩 맡아 동시에 연산
- **lock-free synchronization**: mutex 없이 spin(`_mm_pause`) 기반 handshake로 thread 간 overhead 최소화
- **AVX2 + FMA intrinsics**: `_mm256_fmadd_ps` 등을 직접 사용, prefetch까지 수동 튜닝
- **실측 benchmarking**: 반복 측정 후 average/peak GFLOPS와 결과 검증(`check_result`)까지 내장

## Build & Run

```bash
# GEMM
cd GEMM_SMT/GEMM_SMT
make
./matrix.out

# GEMV
cd GEMV_SMT/GEMV
make
./matrix.out
```

`proc_spec.h`의 `MAIN_CORE` / `SIBLING_CORE`를 실제 physical core–SMT sibling core 번호로 맞춰야 성능 이득이 제대로 납니다(다른 physical core로 잘못 지정하면 거의 2배 성능차가 사라집니다).

## 요구 사항

- x86_64, AVX2 + FMA 지원 CPU (Skylake)
- SMT/Hyper-Threading 활성화
- gcc, Linux(pthread) 기준 — Windows portability layer(`port.h`)도 포함

