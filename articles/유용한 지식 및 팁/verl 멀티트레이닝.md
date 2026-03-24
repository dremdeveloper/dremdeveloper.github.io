# veRL(공식 프로젝트명: `verl`) 멀티 노드 트레이닝 동작 가이드

> 대상: `verl`로 PPO/GRPO 계열 RL post-training을 운영하거나 디버깅해야 하는 엔지니어  
> 톤: 설계/운영 문서 스타일, 배치(batch) 관점 우선  
> 범위: 환경 설정, 단일 노드와의 차이, 멀티 노드 학습 방식, 실행 중 정상성/리소스/OOM 확인

---

## 0. 문서 전제

- 이 문서는 사용자가 표기한 `veRL`을 **공식 프로젝트명 `verl`** 기준으로 설명한다.
- 배치 정규화(normalization) 설명은 **공개 문서 + 공개 소스 기준**으로 정리했다.
- 병렬화 설명은 **기본적으로 FSDP backend 기준**이다.  
  Megatron backend를 쓰면 여기에 **TP/PP(모델 병렬)** 이 추가된다고 보면 된다.
- 문서에 등장하는 **global batch** 는 controller 관점, **local batch** 는 GPU/rank 관점이다.

---

## 1. 한 줄 요약

`verl` 멀티 노드 트레이닝은 **"하나의 controller(드라이버) + 여러 WorkerGroup"** 구조다.

- controller가 한 step의 글로벌 batch를 잡는다.
- `generate_sequences`, `compute_ref_log_prob`, `compute_values`, `update_critic`, `update_actor` 같은 단계는 Ray WorkerGroup으로 RPC 호출된다.
- 각 WorkerGroup은 **자기 `world_size` 기준으로 batch shard를 받아** 계산한다.
- 멀티 노드로 가도 알고리즘 자체는 안 바뀐다.  
  **바뀌는 건 batch를 처리하는 병렬 단위와 inter-node 통신 비용**이다.

---

## 2. 아키텍처 먼저 이해하기

`verl`의 PPO Ray Trainer는 **단일 프로세스 trainer** 가 학습 루프를 orchestration 한다.

대략적인 루프는 아래 순서다.

1. driver가 prompt batch 로드
2. rollout worker group이 응답 생성
3. ref policy가 log prob 계산
4. critic / reward model이 value 또는 reward 계산
5. driver가 advantage 계산
6. actor/critic worker group이 update 수행
7. metric 기록, validation/checkpoint 처리

즉, 멀티 노드라고 해서 "노드별로 독립 학습"을 하는 게 아니다.  
**중앙 루프가 있고, 각 노드는 그 루프가 나눠준 shard를 처리하는 분산 실행체**다.

### 실무 해석

- **driver**
  - 데이터셋에서 step용 batch를 읽음
  - 어떤 role(actor/rollout/ref/critic/reward)이 무엇을 언제 수행할지 결정
  - metric 집계, validation, checkpoint를 관리

- **worker node**
  - rollout / ref / actor / critic 계산을 실제 GPU에서 수행
  - 자기 shard만 처리
  - 분산 전략(FSDP/Megatron)에 따라 gradient / parameter / optimizer state를 동기화

---

## 3. 환경 설정 방법

## 3.1 런타임 전제

멀티 노드에서 가장 중요한 건 **모든 노드가 동일한 런타임을 가진다**는 점이다.

### 권장 기준

- Python >= 3.10
- CUDA >= 12.8
- 동일한 `verl` commit/tag
- 동일한 PyTorch / Ray / NCCL / vLLM(or SGLang) / transformers 버전
- 모든 노드가 동일한 모델 경로에 접근 가능
  - 공유 스토리지
  - 또는 동일한 로컬 경로 사전 배포

### 실무 체크

- [ ] 모든 노드에서 `python -V` 동일
- [ ] 모든 노드에서 `ray --version` 동일
- [ ] 모든 노드에서 `torch.__version__` 동일
- [ ] 모든 노드가 모델/토크나이저 경로를 읽을 수 있음
- [ ] head ↔ worker 간 IP/hostname reachability 확인
- [ ] 방화벽/보안그룹에서 Ray head와 dashboard 포트 허용
- [ ] 가능하면 고속 NIC(IB/RDMA) 사용

> 멀티 노드 FSDP/Megatron에서 네트워크가 느리면, GPU가 남아도 throughput이 안 올라간다.

## 3.2 설치 전략

최신 공개 문서 기준으로는 **Docker 이미지 사용이 제일 안전**하다.

### Docker 권장 이유

- CUDA/cuDNN/PyTorch/vLLM/SGLang 조합을 직접 맞추는 비용 감소
- 노드 간 버전 drift 방지
- 재현성 확보

### Docker 예시

```bash
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" \
  --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl <image:tag> sleep infinity
docker start verl
docker exec -it verl bash
git clone https://github.com/volcengine/verl && cd verl
pip3 install --no-deps -e .
```

### 커스텀 환경 예시

```bash
conda create -n verl python==3.12
conda activate verl

# FSDP 위주
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# source install
git clone https://github.com/volcengine/verl.git
cd verl
pip install --no-deps -e .
```

### 설치 후 반드시 확인할 패키지

- torch 계열
- vLLM
- SGLang
- pyarrow
- tensordict
- (Megatron 사용 시) Apex, 관련 CUDA/cuDNN 패키지

---

## 3.3 Manual Ray Cluster 구성

가장 단순한 멀티 노드 구성은 **head node + worker node** 형태다.

### Head node

```bash
ray start --head --dashboard-host=0.0.0.0
```

여기서 기억할 주소는 2개다.

- **GCS address**: worker가 붙는 주소
- **Dashboard address**: job submit 및 관제 주소 (`http://<head_ip>:8265`)

### Worker node

```bash
ray start --address=<head_gcs_address>
```

### 클러스터 확인

```bash
ray status
```

그리고 브라우저에서:

```text
http://<head_ip>:8265
```

을 열어 dashboard에 들어간다.

---

## 3.4 Job submit 예시

```bash
ray job submit --address="http://<head_ip>:8265" \
  --runtime-env=verl/trainer/runtime_env.yaml \
  --no-wait \
  -- \
  python3 -m verl.trainer.main_ppo \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=2 \
  data.train_files=/data/train.parquet \
  data.val_files=/data/val.parquet \
  ...
```

멀티 노드에서 핵심은 아래 두 값이다.

- `trainer.n_gpus_per_node`
- `trainer.nnodes`

이 값으로 resource pool이 잡히고, WorkerGroup이 **노드 경계를 넘어서** 배치된다.

---

## 4. 단일 노드 트레이닝과 다른 점

| 항목 | 단일 노드 | 멀티 노드 | 실무 포인트 |
|---|---|---|---|
| controller | 1개 | 1개 | 동일 |
| WorkerGroup 배치 | 한 노드 내부 | 여러 노드에 걸쳐 배치 | Ray scheduling 필요 |
| batch 의미 | global/local 구분 필요 | global/local 구분 더 중요 | 설정 해석 실수 방지 |
| 통신 비용 | 주로 intra-node | inter-node 포함 | NIC/IB 품질 중요 |
| 디버깅 범위 | 프로세스/노드 | 클러스터 전체 | dashboard 필수 |
| 장애 양상 | CUDA OOM, local deadlock | + Ray actor death, NIC 병목, host OOM | 관측 포인트 증가 |

### 핵심 메시지

멀티 노드는 **학습 알고리즘을 바꾸는 기능이 아니라, 같은 학습 루프를 더 큰 분산 자원에 매핑하는 기능**이다.

즉:

- `train_batch_size` 의미 그대로
- `ppo_mini_batch_size` 의미 그대로
- `micro_batch_size_per_gpu` 의미 그대로

다만, **이 값들이 내부적으로 worker/rank 기준으로 normalize** 된다는 점을 정확히 이해해야 한다.

---

## 5. 멀티 노드 트레이닝 방법 — batch 관점 설명

이 섹션이 핵심이다.

## 5.1 배치 파라미터를 먼저 분류하자

### 알고리즘 관점의 global batch

- `data.train_batch_size`
  - 한 training iteration에서 샘플링하는 **prompt 수**
- `actor_rollout_ref.rollout.n`
  - prompt당 생성할 **response 수**
- `actor_rollout_ref.actor.ppo_mini_batch_size`
  - actor update용 **global mini-batch 크기**
- `critic.ppo_mini_batch_size`
  - critic update용 **global mini-batch 크기**

### 성능/메모리 관점의 local batch

- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`
- `critic.ppo_micro_batch_size_per_gpu`
- `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`
- `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`

이 값들은 **한 GPU가 한 번에 처리하는 양**이다.  
즉, throughput/OOM에는 직접 영향이 크지만 **알고리즘 수렴 의미를 바꾸는 값은 아니다.**

---

## 5.2 한 step에서 실제로 벌어지는 일

아래 설정을 가정하자.

```yaml
data.train_batch_size: 1024
actor_rollout_ref.rollout.n: 4
actor_rollout_ref.actor.ppo_mini_batch_size: 256
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu: 8
trainer.nnodes: 2
trainer.n_gpus_per_node: 8
```

### Step A. driver가 글로벌 prompt batch를 로드한다

한 step 시작 시점의 prompt 수는:

```text
1024 prompts
```

이다.

### Step B. rollout이 trajectory를 만든다

PPO 문서 기준으로 전체 trajectory 수는:

```text
global_trajectories = data.train_batch_size * rollout.n
                    = 1024 * 4
                    = 4096
```

즉, step 하나에서 실제 update 대상으로 들어가는 응답/trajectory는 **4096개**다.

### Step C. driver가 batch를 WorkerGroup에 분배한다

`generate_sequences` 는 `Dispatch.DP_COMPUTE_PROTO` 로 등록되어 있고,  
이 dispatch는 `DataProto.chunk` 를 사용해 큰 batch를 **`worker_group.world_size` 개로 분할**한다.  
여기서 `world_size` 는 **노드 수가 아니라, 그 WorkerGroup의 실제 분산 실행 단위 수**다.

예를 들어 actor/rollout WorkerGroup의 유효 DP world size가 16이면:

```text
4096 trajectories / 16 ranks = rank당 256 trajectories
```

가 된다.

### Step D. 각 rank는 local shard만 계산한다

각 rank/GPU는 자기에게 온 local shard만 가지고:

- rollout 생성
- ref logprob 계산
- value 계산
- actor/critic update

를 수행한다.

### Step E. update 단계에서는 mini-batch와 micro-batch가 다시 갈라진다

글로벌 actor mini-batch가 256이면, prompt 기준으로는 256개지만  
actor FSDP worker의 normalize 로직은 내부적으로 `rollout.n` 을 반영한다.

즉, actor 쪽 local mini-batch 감각은 아래처럼 잡으면 된다.

```text
local_actor_mini_batch
  ≈ actor.ppo_mini_batch_size * rollout.n / effective_dp_size
```

유효 DP size가 16이라면:

```text
local_actor_mini_batch = 256 * 4 / 16 = 64 trajectories
```

그리고 micro batch가 8이면:

```text
grad_accum_steps = 64 / 8 = 8
```

즉, 각 GPU는 **8 trajectories씩 8번** forward/backward를 수행해서 local mini-batch를 소화한다.

> 여기서 `effective_dp_size` 는 FSDP device mesh와 Ulysses sequence parallel 크기를 반영한 값이다.

---

## 5.3 어떻게 안 겹치나?

질문의 핵심이 이 부분이다.

### 같은 step 안에서는 안 겹친다

이유는 단순하다.

1. driver가 글로벌 batch를 하나 만든다.
2. `DP_COMPUTE_PROTO` dispatch가 이를 `world_size` 기준으로 **chunk** 한다.
3. 각 chunk는 **하나의 worker/rank** 로만 전달된다.
4. collect 단계에서 다시 합친다.

즉, **동일 step 안에서 같은 샘플 row가 여러 worker에 중복 배분되지 않도록 중앙에서 먼저 잘라서 보낸다.**

### node 기준이 아니라 world_size 기준이다

중요한 포인트는 "2노드니까 반반"이 아니라:

```text
해당 WorkerGroup의 world_size 기준으로 shard
```

라는 점이다.

예를 들어:

- 2 nodes × 8 GPUs
- rollout TP=1
- effective DP=16

이면 16 shard로 자른다.

반대로 Megatron에서 TP/PP가 섞이면, **데이터 분배의 기준은 DP 축**이고 TP/PP rank는 같은 shard를 협업 처리한다.

### 딱 안 나눠떨어지면?

`DP_COMPUTE_PROTO` dispatch는 **auto padding** 으로 split을 맞춘다.

즉:

- batch가 world_size로 정확히 안 나눠져도 dispatch 자체는 처리 가능
- 그래도 운영/성능 해석을 쉽게 하려면 아래 값은 되도록 **유효 DP size로 나눠떨어지게** 잡는 걸 권장한다

```text
data.train_batch_size
data.val_batch_size
actor/critic ppo_mini_batch_size
```

이렇게 해두면:

- padding/경계 조건 디버깅이 쉬워지고
- per-rank shard 크기 해석이 쉬워지고
- stall이나 imbalance 원인도 더 빨리 찾을 수 있다

---

## 5.4 각 노드는 어떤 방식으로 학습하나?

## rollout phase

각 노드는 자기 shard prompt만 받는다.

- rollout engine(vLLM/SGLang/HF)이 응답 생성
- old logprob 또는 관련 생성 결과를 반환
- 노드 간에 "서로 다른 샘플"을 처리

즉, rollout은 **데이터 병렬 관점에서 서로 다른 샘플을 분산 생성**한다고 이해하면 된다.

## actor/critic update phase (FSDP 기준)

각 rank/GPU는 자기 local micro-batch로 forward/backward를 수행한다.  
그 뒤 FSDP가 아래를 담당한다.

- parameter shard 관리
- gradient 동기화
- optimizer state shard 관리
- logical model consistency 유지

즉, 멀티 노드라고 해서 **노드별로 서로 다른 모델을 따로 학습**하는 게 아니다.

정확한 표현은 이거다.

```text
같은 모델을 여러 rank가 shard 형태로 나눠 들고
서로 다른 데이터 shard를 처리한 뒤
동기화해서 하나의 global update를 만든다
```

## Megatron을 쓰는 경우

Megatron backend에서는 3D parallel이 추가될 수 있다.

- **DP**: 서로 다른 data shard 처리
- **TP/PP**: 같은 data shard를 모델 병렬로 협업 처리

즉, "샘플이 안 겹친다"는 질문의 답은 **DP 차원에서 안 겹친다**가 정확하다.

---

## 5.5 숫자로 보는 전체 예시

다음 구성을 가정한다.

```yaml
trainer.nnodes: 2
trainer.n_gpus_per_node: 8

data.train_batch_size: 1024
actor_rollout_ref.rollout.n: 4

actor_rollout_ref.actor.ppo_mini_batch_size: 256
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu: 8
```

### 글로벌 기준

- 총 GPU: 16
- 한 step prompt 수: 1024
- 한 step trajectory 수: 4096

### rollout 분배

effective DP size = 16 이면:

```text
rank당 rollout shard = 4096 / 16 = 256 trajectories
```

### actor mini-batch 분배

global actor mini-batch = 256 prompts  
trajectory 기준으로 환산하면:

```text
256 * 4 = 1024 trajectories
```

local actor mini-batch는:

```text
1024 / 16 = 64 trajectories
```

micro batch가 8이면:

```text
64 / 8 = 8 micro-steps
```

### step당 actor mini-batch 개수

prompt 기준 global step에서 actor mini-batch 개수는:

```text
1024 / 256 = 4 mini-batches
```

즉, step 하나에서:

- actor update는 global mini-batch 4개를 순회하고
- 각 mini-batch 안에서 각 GPU는 8번 accumulation 한다

이 구조를 이해하면 **throughput / GPU utilization / OOM 원인**이 훨씬 빨리 보인다.

---

## 6. 트레이닝 중 세부 동작 확인 방법

## 6.1 정상성 확인: 제일 먼저 보는 것

### 1) Ray cluster 상태

```bash
ray status
```

확인 포인트:

- 모든 노드가 붙어 있는가
- 예상한 GPU 수가 잡혔는가
- pending resource가 계속 쌓이는가
- 특정 노드만 비정상적으로 유휴 상태인가

### 2) Job 상태

```bash
ray job list
ray job status <submission_id>
ray job logs <submission_id>
ray job logs <submission_id> --follow
```

### 3) Ray 로그 디렉터리

```text
/tmp/ray/session_latest/logs/
```

특히 driver 로그:

```text
job-driver-raysubmit_<submission_id>.log
```

멀티 노드에선 여기서:

- role init 실패
- actor death
- worker restart
- scheduling mismatch

를 가장 빨리 발견한다.

### 4) Dashboard

```text
http://<head_ip>:8265
```

멀티 노드에서는 dashboard를 보는 걸 사실상 필수로 생각하는 게 좋다.

- 어떤 actor가 어느 노드에 있는지
- 어떤 task가 오래 걸리는지
- resource가 어디에 몰려 있는지
- worker가 죽고 다시 뜨는지

를 구조적으로 볼 수 있다.

---

## 6.2 학습 루프 정상성 확인

PPO Trainer 기본 루프는 단계별 timing metric을 기록한다.

대표적으로:

- `timing/gen`
- `timing/ref`
- `timing/values`
- `timing/adv`
- `timing/update_critic`
- `timing/update_actor`
- `timing/testing` (validation 켠 경우)

### 정상 패턴

- step마다 위 timing이 계속 전진
- actor/critic metric이 지속적으로 갱신
- validation을 켰다면 `test_freq` 주기마다 validation metric이 들어옴
- 특정 단계의 latency가 갑자기 수배 이상 튀지 않음

### 비정상 패턴

- `timing/gen` 에서 오래 멈춤  
  → rollout engine / rollout worker / serving side 리소스 문제 가능성 높음
- `timing/update_actor` 또는 `timing/update_critic` 에서 멈춤  
  → collective/NCCL, micro-batch 설정, OOM, FSDP 문제 가능성 높음
- step은 가는데 throughput이 급락  
  → long-tail rollout, inter-node 네트워크 병목, 지나치게 작은 micro-batch 가능성 높음

---

## 6.3 리소스 점유 정도 확인

## 6.3.1 GPU

```bash
watch -n1 nvidia-smi
nvidia-smi dmon -s pucvmet
```

보는 포인트:

- GPU memory가 step 진행 중 안정적으로 유지되는가
- 특정 GPU만 memory/util이 튀는가
- util이 0에 가까운데 프로세스만 살아 있는가
- rollout 쪽과 actor/critic 쪽 사용 패턴이 비정상적으로 벌어지는가

## 6.3.2 Host RAM / CPU / Disk

```bash
htop
free -h
df -h
iostat -xz 1
```

보는 포인트:

- host RAM이 step마다 우상향하는가
- swap 사용 여부
- `/tmp/ray` 또는 로그 디스크가 차는가
- parquet 로딩/로그 flush 때문에 I/O 병목이 생기는가

## 6.3.3 Network

```bash
sar -n DEV 1
iftop
```

멀티 노드에서 update 단계가 느려지면 GPU만 보지 말고 네트워크도 같이 봐야 한다.  
특히 actor/critic collective가 inter-node를 많이 타면 NIC saturation이 throughput ceiling이 된다.

---

## 6.4 OOM 발생 가능성 확인과 대응

OOM은 **GPU OOM**, **host RAM OOM**, **Ray memory monitor에 의한 worker kill** 로 나눠서 봐야 한다.

## 6.4.1 자주 보이는 OOM/메모리 이상 징후

### GPU OOM

- `CUDA out of memory`
- 특정 step에서만 actor/critic update 직전 또는 직후 죽음
- 긴 sequence 샘플 비율이 높은 step에서만 재현

### Host/Ray OOM

- Ray actor가 갑자기 죽음
- 로그에 아래 류 메시지 출력

```text
Worker exit type: SYSTEM_ERROR
Worker unexpectedly exits with a connection error code 2.
The process is killed by SIGKILL by OOM killer due to high memory usage.
```

이 패턴은 "코드 버그"일 수도 있지만, 실제 운영에서는 **host OOM / Ray OOM kill** 로 먼저 의심하는 게 맞다.

### 확인 명령

```bash
dmesg -T | egrep -i 'killed process|out of memory|oom'
```

그리고 Ray 로그와 `/tmp/ray/session_latest/logs/` 를 함께 본다.

---

## 6.4.2 OOM 대응 우선순위

### 1순위: micro batch 줄이기

가장 먼저 줄일 값:

- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`
- `critic.ppo_micro_batch_size_per_gpu`
- `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`
- `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`

원칙은 단순하다.

> **알고리즘 batch(global)는 유지하고, local micro-batch를 줄여서 살린다.**

### 2순위: rollout 메모리 조절

rollout 쪽에서 먼저 죽으면:

- `actor_rollout_ref.rollout.gpu_memory_utilization` 낮추기
- `max_num_batched_tokens` / `max_num_seqs` 조절
- `rollout.n` 줄이기

특히 `gpu_memory_utilization` 은 너무 높이면 OOM 확률이 크게 오른다.  
문서 기준으로는 0.5~0.7이 대체로 균형이 좋다고 안내한다.

### 3순위: gradient checkpointing / activation offload

```yaml
actor_rollout_ref.model.enable_gradient_checkpointing: True
critic.model.enable_gradient_checkpointing: True
actor_rollout_ref.model.enable_activation_offload: True
critic.model.enable_activation_offload: True
```

- 대형 모델
- 긴 sequence
- 큰 mini-batch

에서는 거의 기본 카드다.

### 4순위: dynamic batch size 전환

```yaml
actor.use_dynamic_bsz: True
critic.use_dynamic_bsz: True
ref.log_prob_use_dynamic_bsz: True
rollout.log_prob_use_dynamic_bsz: True
```

이 경우 `*micro_batch_size_per_gpu` 대신:

- `ppo_max_token_len_per_gpu`
- `log_prob_max_token_len_per_gpu`
- `forward_micro_batch_size_per_gpu`

쪽을 튜닝한다.

### 5순위: long context 대응

32k 이상 long sequence 학습이면:

- `*micro_batch_size_per_gpu` 감소
- `*max_token_len_per_gpu` 감소
- 필요 시 `ulysses_sequence_parallel_size > 1`

이 순서로 본다.

---

## 6.4.3 actor/critic/reward batch 크기를 똑같이 맞출 필요는 없다

`verl` 문서의 튜닝 가이드 기준으로:

- forward-only 계열(`ref.log_prob_micro_batch_size_per_gpu`, `rollout.log_prob_micro_batch_size_per_gpu`)은
  actor training micro-batch보다 더 크게 잡을 수 있다.
- critic/reward도 actor보다 더 크게 잡을 수 있는 경우가 많다.

이유는 단순하다.

- actor는 최종 vocab projection 때문에 메모리 부담이 크고
- critic/reward는 상대적으로 덜 무거운 경우가 많다

즉, **"전 role micro-batch를 동일하게 맞춘다"** 는 접근은 보통 비효율적이다.

---

## 6.5 좀 더 깊게 보는 방법

## 6.5.1 Ray Distributed Debugger

최신 Ray Distributed Debugger VSCode extension 사용이 권장된다.

### 핵심

- dashboard URL로 cluster attach
- remote code 안에 `breakpoint()` 삽입
- `RAY_DEBUG_POST_MORTEM=1` 설정 가능

```bash
export RAY_DEBUG_POST_MORTEM=1
```

> breakpoint는 `@ray.remote` 로 실행되는 함수 영역에서 동작하는 점을 기억하면 된다.

## 6.5.2 Ray timeline

학습 job 끝에 timeline JSON을 남기면 병목 분석이 쉬워진다.

```yaml
ray_init.timeline_json_file=/tmp/ray_timeline.json
```

생성된 파일은:

- `chrome://tracing`
- Perfetto UI

로 열어서 task 간 공백과 병목 구간을 볼 수 있다.

## 6.5.3 verl profiler system

더 깊은 성능 분석이 필요하면 `global_profiler` 를 켠다.

대표 설정 개념:

- `global_profiler.tool`
- `global_profiler.steps`
- `global_profiler.save_path`
- role별 `profiler.enable`
- role별 `profiler.all_ranks` / `profiler.ranks`

즉, **"몇 step을, 어떤 role을, 어느 rank에서 프로파일링할지"** 를 분리해서 제어할 수 있다.

## 6.5.4 Nsight / PyTorch profiler

### Nsight

- `global_profiler.steps` 로 특정 step만 수집
- 결과는 기본적으로 각 노드의 `/tmp/ray/session_latest/logs/nsight/` 아래 저장

### PyTorch profiler

- `global_profiler.steps`
- `global_profiler.save_path`
- role별 `enable`, `all_ranks`, `ranks`
- `contents: [cpu, cuda, memory, shapes, stack]`

까지 세밀하게 켤 수 있다.

---

## 6.6 Prometheus / Grafana로 rollout 상태 보기

rollout 상태를 별도로 보고 싶으면 Prometheus/Grafana 관제를 붙일 수 있다.

### 핵심 설정

```yaml
actor_rollout_ref.rollout.disable_log_stats: False
actor_rollout_ref.rollout.prometheus.enable: True
```

### 확인 포인트

- Prometheus: `http://master_ip:9090`
  - `vllm:`
  - `sglang:`
- Grafana: `http://master_ip:3000`

이 방법은 특히 **rollout serving throughput**, **cache hit rate**, **engine metric** 을 보고 싶을 때 유용하다.

---

## 7. 운영 체크리스트

## 학습 시작 전

- [ ] 모든 노드 런타임 버전 동일
- [ ] Ray cluster 정상
- [ ] dashboard 접속 가능
- [ ] 모델/데이터 경로 접근 가능
- [ ] `trainer.nnodes`, `trainer.n_gpus_per_node` 값 확인
- [ ] `train_batch_size`, `ppo_mini_batch_size` 가 유효 DP size와 잘 맞는지 확인
- [ ] rollout `gpu_memory_utilization` 과 actor/critic micro-batch가 보수적으로 시작되었는지 확인

## 학습 중

- [ ] `timing/gen`, `timing/ref`, `timing/values`, `timing/update_actor`, `timing/update_critic` 전진 여부 확인
- [ ] `nvidia-smi` 로 node/GPU별 memory 균형 확인
- [ ] `/tmp/ray/session_latest/logs/` 에 actor death/OOM 메시지 없는지 확인
- [ ] validation 주기와 checkpoint 주기 정상 동작 확인
- [ ] throughput 급락 시 network/NCCL/rollout queue 병목 확인

## 장애 시 1차 조치

1. `ray job logs <id> --follow`
2. `tail -f /tmp/ray/session_latest/logs/job-driver-raysubmit_<id>.log`
3. `watch -n1 nvidia-smi`
4. `free -h`, `dmesg -T | grep -i oom`
5. `ppo_micro_batch_size_per_gpu` / `log_prob_micro_batch_size_per_gpu` 우선 감축
6. 필요 시 `gpu_memory_utilization` 하향
7. long sequence면 `max_prompt_length`, `max_response_length`, `*_max_token_len_per_gpu` 하향
8. 재현이 어려우면 timeline / profiler 켜기

---

## 8. 최종 정리

`verl` 멀티 노드 학습을 이해할 때는 아래 4문장만 정확히 잡으면 된다.

1. **controller는 하나고, WorkerGroup이 분산 계산을 한다.**
2. **`train_batch_size` 와 `ppo_mini_batch_size` 는 global 값이다.**
3. **`*micro_batch_size_per_gpu` 는 local 값이고, 성능/OOM을 제어한다.**
4. **동일 step 안에서 샘플이 안 겹치는 이유는 driver가 batch를 `world_size` 기준으로 chunk 해서 shard를 배분하기 때문이다.**

즉, 멀티 노드 운영의 본질은 이거다.

```text
global batch를 어떻게 local shard로 안전하게 분해하고,
그 local shard를 어떤 micro-batch로 태워서,
OOM 없이 높은 utilization로 유지할 것인가
```

이 관점으로 보면, 대부분의 문제는 아래 셋 중 하나로 분해된다.

- batch 의미를 잘못 이해한 설정 문제
- 분산 통신/배치 균형 문제
- 메모리 예산(local micro-batch / token budget) 문제

---

## 참고 자료

- [R1] Installation  
  https://verl.readthedocs.io/en/latest/start/install.html

- [R2] Multinode Training  
  https://verl.readthedocs.io/en/latest/start/multinode.html

- [R3] PPO Ray Trainer  
  https://verl.readthedocs.io/en/latest/workers/ray_trainer.html

- [R4] The Design of `verl.single_controller`  
  https://verl.readthedocs.io/en/latest/single_controller.html

- [R5] Config Explanation  
  https://verl.readthedocs.io/en/latest/examples/config.html

- [R6] PPO  
  https://verl.readthedocs.io/en/latest/algo/ppo.html

- [R7] Performance Tuning Guide  
  https://verl.readthedocs.io/en/latest/perf/perf_tuning.html

- [R8] Ray Debug Tutorial  
  https://verl.readthedocs.io/en/latest/start/ray_debug_tutorial.html

- [R9] verl Profiler System  
  https://verl.readthedocs.io/en/latest/perf/verl_profiler_system.html

- [R10] NVIDIA Nsight Systems profiling in verl  
  https://verl.readthedocs.io/en/latest/perf/nsight_profiling.html

- [R11] PyTorch Profiling in verl  
  https://verl.readthedocs.io/en/latest/perf/torch_profiling.html

- [R12] Use Prometheus and Grafana to Monitor Rollout  
  https://verl.readthedocs.io/en/latest/advance/grafana_prometheus.html

- [R13] FSDP worker normalization code  
  https://github.com/verl-project/verl/blob/main/verl/workers/fsdp_workers.py

- [R14] OOM/actor death 패턴 예시  
  https://github.com/verl-project/verl/issues/472
