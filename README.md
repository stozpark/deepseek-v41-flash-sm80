# DeepSeek-V4.1-Flash on NVIDIA A100/A800 (SM80)

`deepseek-ai/DeepSeek-V4.1-Flash`를 **NVIDIA A100/A800 (SM80)** 에서 vLLM으로
서빙하기 위한 재현 가능한 실행/패치 패키지입니다.

핵심 원칙은 공식 체크포인트를 변환하지 않고, Ampere 지원이 들어간
`wtdcode/vllm-backport`의 **v0.13.0 SM80 build**를 고정해서 사용하는 것입니다.

> 이 저장소는 vLLM 전체를 복제하지 않습니다. A100용 runtime pin, Singularity
> recipe, launcher, preflight/verification script와 패치 계보를 제공합니다.

## 기준 버전

```text
Model:   deepseek-ai/DeepSeek-V4.1-Flash (official weights)
Runtime: wtdcode/vllm-backport
Commit:  24cb31bb4fd0becee65c810c913a8caa4f610c36
Image:   lazymio/vllm-backport:v0.13.0-sm80
GPU:     NVIDIA A100/A800, SM80
CUDA:    image is CUDA 13.x based
```

A100에서 중요한 변경은 다음입니다.

- Hopper 전용 sparse MLA 경로 대신 **SM8x Triton sparse-MLA** 사용
- SM80에서 직접 지원되지 않는 FP8 E4M3 변환을 **software encode/decode**로 처리
- V4.1 indexer의 128-token page layout 지원
- Engram table을 **CPU offload**
- DSpark-5 speculative decoding + local argmax reduction

세부 커밋은 [`patches/SM80_PATCHSET.md`](patches/SM80_PATCHSET.md)에 정리했습니다.

## 권장 하드웨어

기본 recipe는 다음을 가정합니다.

```text
GPU:       A100 80GB x 8 (TP8)
Host RAM:  256 GiB 최소권장, 384 GiB+ 여유 권장
Disk:      모델 + 이미지 저장 여유 공간 충분히 확보
Driver:    CUDA 13.x를 직접 사용할 경우 R580+ 권장
```

V4.1 Engram CPU offload 자체가 약 196 GB를 사용하므로 256 GiB는 여유가 크지 않습니다.
가능하면 384 GiB 이상을 권장합니다. 모델 weight는 약 320 GB 규모로 보고된 구성입니다.

## 1. 사전 점검

```bash
./preflight.sh
```

특히 driver가 R580 미만이면 CUDA 13.x 호환성 경고가 나옵니다. NVIDIA의
`cuda-compat-13-0` forward-compatibility 경로를 사용할 수 있는 환경인지 확인하거나
driver를 올리는 것이 안전합니다.

## 2-A. Singularity/Apptainer 이미지 생성

```bash
./build_sif.sh
```

기본 출력:

```text
deepseek-v41-flash-sm80-cu130.sif
```

fakeroot를 쓰지 않는 환경이라면:

```bash
BUILD_ARGS="" ./build_sif.sh
```

단, 시스템 정책에 따라 root 권한이 필요할 수 있습니다.

## 2-B. Docker를 바로 쓰는 경우

```bash
MODEL_PATH=/path/to/DeepSeek-V4.1-Flash \
./serve_docker_a100.sh
```

## 3. A100 x8 권장 실행

오프라인/로컬 checkpoint:

```bash
MODEL_PATH=/models/DeepSeek-V4.1-Flash \
SIF_PATH=./deepseek-v41-flash-sm80-cu130.sif \
./serve_tp8_a100.sh
```

Hugging Face에서 직접 읽을 수 있는 환경:

```bash
MODEL_PATH=deepseek-ai/DeepSeek-V4.1-Flash \
./serve_tp8_a100.sh
```

기본값은 **검증 보수값**으로 256K context를 사용합니다.

```text
TP                         8
max_model_len              262144
max_num_seqs               16
max_num_batched_tokens     16384
gpu_memory_utilization     0.90
KV cache                    fp8_ds_mla
Engram                      CPU offload
Prefix cache                on
DSpark                      5 tokens
CUDA graph                  FULL_AND_PIECEWISE
Custom all-reduce           off
NCCL                        Ring / Simple
```

변경 예:

```bash
MAX_MODEL_LEN=524288 \
GPU_MEMORY_UTILIZATION=0.95 \
MODEL_PATH=/models/DeepSeek-V4.1-Flash \
./serve_tp8_a100.sh
```

1M context는 모델이 지원하더라도 A100에서 바로 기본값으로 잡지 않았습니다.
256K -> 512K -> 1M 순서로 KV cache 여유와 long-context correctness를 확인하는 것을
권장합니다.

## 4. Expert Parallel

backport는 EP도 지원합니다. 먼저 TP8 baseline을 확인한 뒤 켜는 것이 좋습니다.

```bash
ENABLE_EXPERT_PARALLEL=1 \
MODEL_PATH=/models/DeepSeek-V4.1-Flash \
./serve_tp8_a100.sh
```

## 5. DSpark 끄기

문제 분리를 위해 speculative decoding 없이 시작하려면:

```bash
DISABLE_DSPARK=1 \
MODEL_PATH=/models/DeepSeek-V4.1-Flash \
./serve_tp8_a100.sh
```

baseline이 뜬 뒤 DSpark를 다시 켜십시오. A100/NVLink 환경에서는 DSpark가 decode
속도에 큰 영향을 줄 수 있습니다.

## 6. Smoke test

서버가 올라온 뒤:

```bash
./smoke_test.sh
```

또는:

```bash
curl http://127.0.0.1:18005/v1/models
```

## 7. Runtime 내부 검증

컨테이너 안에서:

```bash
apptainer exec --nv deepseek-v41-flash-sm80-cu130.sif \
  python3 verify_runtime.py
```

`vllm/models/deepseek_v4_1/ampere/ampere_sparse.py`가 존재하는지와 GPU compute
capability가 SM80인지 확인합니다.

## A100에서 왜 별도 경로가 필요한가

DeepSeek-V4.1의 최신 고성능 경로에는 Hopper 이후 GPU에 최적화된 FP8/sparse MLA
구성이 포함됩니다. A100(SM80)은 그 경로를 그대로 실행할 수 없습니다.

Ampere port는 크게 다음을 바꿉니다.

```text
Sparse MLA     Hopper kernel     -> Triton SM8x sparse MLA
FP8 cache I/O  native E4M3       -> software E4M3 encode/decode
Engram         GPU-resident      -> CPU offload
DSpark         generic path      -> V4.1 shard/local-argmax hooks
```

즉 모델 아키텍처나 checkpoint를 바꾸는 것이 아니라 **runtime backend를 Ampere에
맞게 교체**합니다.

## 검증된 공개 결과 기준

8 x A100에서 다음 설정으로 동작 보고가 있습니다.

```text
TP8
max_model_len=262144
max_num_seqs=16
gpu_memory_utilization=0.95
kv_cache_dtype=fp8_ds_mla
Engram CPU offload
FULL_AND_PIECEWISE CUDA graph
commit=24cb31b
```

보고된 decode는 DSpark 설정 전후에 차이가 있으므로 숫자를 절대값으로 보지 마십시오.
초기 측정은 단일 스트림 약 34.8 tok/s였고, DSpark를 활성화한 NVLink A100에서는
약 90 tok/s가 보고되었습니다. topology와 acceptance rate의 영향이 큽니다.

## Source checkout가 필요한 경우

```bash
./prepare_patched_source.sh
```
