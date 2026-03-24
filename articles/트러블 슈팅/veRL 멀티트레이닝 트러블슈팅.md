# veRL 멀티트레이닝 트러블슈팅

> 작성 기준: 2026-03-24 기준 공개된 **공식 저장소 / 공식 문서 / 공식 GitHub 이슈**를 바탕으로 정리한 실무형 가이드입니다.  
> 범위: **멀티 GPU / 멀티 노드 / Ray + FSDP 또는 Megatron + vLLM 또는 SGLang** 환경에서의 RL post-training(PPO/GRPO/DAPO 포함) 문제 대응.  
> 표기: `verl`은 프로젝트 공식 표기, 본 문서에서는 사용자가 요청한 표현을 반영해 **veRL(verl)** 로 함께 적습니다.

---

## 0. 가장 먼저 보는 60초 체크리스트

멀티트레이닝이 막힐 때는 아래 7가지를 먼저 확인하세요.

1. **Ray 클러스터가 실제 노드 수를 모두 보고 있는가?**  
   - `ray status`
   - Ray dashboard / worker 등록 여부

2. **GPU/CPU/메모리/SHM가 실제 자원과 맞는가?**  
   - `nvidia-smi`
   - `free -h`
   - `df -h /dev/shm`
   - `ulimit -n`

3. **분산 설정이 실제 자원 수와 일치하는가?**  
   - `trainer.nnodes`
   - `trainer.n_gpus_per_node`
   - `actor_rollout_ref.rollout.tensor_model_parallel_size`
   - (fully async면) trainer/rollouter의 노드, GPU 개수 분리 설정

4. **길이와 샘플 수가 과격하지 않은가?**  
   - `data.max_prompt_length`
   - `data.max_response_length`
   - `actor_rollout_ref.rollout.n`
   - 멀티턴이면 turn 수 제한

5. **메모리 안전 모드로 시작했는가?**  
   - `enable_gradient_checkpointing=True`
   - 필요한 경우 `enable_activation_offload=True`
   - FSDP 대형 모델이면 `dtensor` 계열 로더 우선
   - `gpu_memory_utilization` 낮춰 시작

6. **SLURM/컨테이너의 CPU 제한을 Ray가 모르고 있지 않은가?**  
   - SLURM이면 `ray_init.num_cpus`를 명시
   - 컨테이너 cgroup 제한과 Ray의 CPU 인식이 일치하는지 확인

7. **현재 backend 조합이 “공식적으로 가장 안정적인 조합”인가?**  
   - 문서 기준으로 fully async는 **현재 `megatron/fsdp + vllm`** 조합이 지원 모드입니다.
   - SGLang rollout은 공식 문서에서 **extensive development** 상태로 소개됩니다.

---

## 1. veRL 멀티트레이닝에서 장애가 나는 위치

veRL의 분산 학습 장애는 보통 한 군데가 아니라 아래 층위 중 하나에서 발생합니다.

1. **스케줄링 층**: Ray worker 등록, raylet heartbeat, actor 생성/등록 실패  
2. **분산 통신 층**: NCCL/RCCL, NIC 선택, IP/포트/방화벽, 멀티노드 접속 실패  
3. **호스트 자원 층**: CPU oversubscription, DRAM 부족, `/dev/shm` 부족, FD(파일 디스크립터) 부족  
4. **GPU 자원 층**: rollout KV cache, FSDP all-gather, FULL_STATE_DICT sync, 긴 sequence로 인한 OOM  
5. **설정 층**: world size 불일치, batch/length/TP/SP 파라미터 충돌, async 모드 조합 불일치  
6. **버전/호환성 층**: vLLM/SGLang/Ray/Torch/grpcio 조합 문제  
7. **체크포인트 층**: 저장 속도, 저장 실패, resume world size 불일치

이 때문에 **증상만 보고 한 번에 결론 내리면 오진**하기 쉽습니다.  
예를 들어 `Failed to register worker to Raylet`는 “Ray 문제”처럼 보이지만, 실제로는 **SLURM CPU 제한**, **agent heartbeat 실패**, **OOM**, **grpcio/포트 충돌**까지 넓게 엮일 수 있습니다.

---

## 2. 자원 이슈(cpu, memory, shm, GPU, network)

## 2.1 CPU oversubscription / SLURM 제약 / Raylet 등록 실패

### 대표 증상
- `Failed to register worker to Raylet`
- worker가 생성되다 끊기고 `End of file`
- node marked dead / missed too many heartbeats
- Ray는 떠 있지만 학습 시작 직후 actor 생성이 불안정함

### 공식 근거에서 보이는 원인
공식 FAQ는 `Unable to register worker with raylet`의 주요 원인으로 **시스템 설정, 특히 SLURM의 CPU 공유 제약**을 직접 언급합니다. Ray가 머신의 CPU 코어 수만큼 worker를 띄우려 하지만, SLURM 제약 때문에 worker가 raylet을 보지 못해 문제가 난다고 설명합니다. 따라서 **`ray_init.num_cpus`를 시스템이 허용하는 값으로 명시**하라고 안내합니다.  
또한 공식 이슈에서는 같은 증상 뒤에 **missed heartbeats**, **agent failure**, **grpcio 버전**, **포트 충돌**, **OOM 가능성**이 이어집니다.

### 확인 포인트
```bash
echo "$SLURM_CPUS_PER_TASK"
echo "$SLURM_CPUS_ON_NODE"
ray status
ps -ef | grep raylet
top -H
```

### 조치 우선순위
1. **SLURM 환경이면 `ray_init.num_cpus`를 명시**합니다.
2. Ray worker가 사용할 CPU와 **driver/raylet/agent가 쓸 CPU를 분리**합니다.
3. 컨테이너의 cgroup CPU limit, SLURM `cpus-per-task`, Ray `num_cpus`가 서로 맞는지 확인합니다.
4. 증상이 heartbeat로 이어지면:
   - `grpcio` 버전
   - agent 포트 충돌
   - OOM killer 흔적 (`dmesg`, `journalctl -k`)
   - Ray agent 로그
   를 같이 봅니다.
5. CPU를 너무 넉넉히 주지 못하는 클러스터에서는 **초기 actor 수를 줄여 최소 구성으로 먼저 기동**합니다.

### 실무 팁
- “CPU는 남아도는데 왜 raylet이 죽지?”가 아니라, **Ray가 “보는 CPU 수”와 SLURM이 “허용한 CPU 수”가 다르면** 죽을 수 있습니다.
- 이 계열 장애는 GPU OOM처럼 보이지 않아도 결국 **시스템 제약 문제**인 경우가 많습니다.

---

## 2.2 호스트 메모리(DRAM) 부족 / 데이터셋 적재 / SIGBUS

### 대표 증상
- `Ray SIGBUS error on my dataset`
- `The actor is dead because its worker process has died`
- `Worker exit detail: ... OOM killer due to high memory usage`
- 시작은 되지만 데이터 로드/직렬화 구간에서 갑자기 죽음

### 공식 근거에서 보이는 원인
공식 config 문서는 **`data.train_files`를 모두 메모리로 읽기 때문에 너무 크면 안 되며(<100GB)**, 로컬/HDFS 경로 모두 DRAM 사용을 유발할 수 있다고 설명합니다.  
공식 이슈 #3218은 `SIGBUS`와 함께 **OOM killer / unexpected crash / connection EOF** 가능성을 직접 보여줍니다.

### 확인 포인트
```bash
free -h
vmstat 1
dmesg | egrep -i "killed process|oom|sigbus"
ls -lh <train.parquet> <val.parquet>
```

### 조치 우선순위
1. 데이터셋을 **더 작은 parquet shard**로 분할합니다.
2. dry run에서는:
   - `data.train_max_samples`
   - `data.val_max_samples`
   를 줄여 최소 재현부터 합니다.
3. 길이와 샘플 수를 낮춥니다.
   - `data.max_prompt_length`
   - `data.max_response_length`
   - `actor_rollout_ref.rollout.n`
4. 긴 샘플이 문제라면:
   - `data.filter_overlong_prompts=True`
   - 필요 시 `data.filter_overlong_prompts_workers` 증가
5. 멀티모달이면 image payload/직렬화 부하를 줄입니다.

### 실무 팁
- veRL에서는 “GPU 메모리는 괜찮은데 죽는다”가 종종 **호스트 DRAM 문제**입니다.
- 특히 멀티턴/멀티모달/긴 reasoning 데이터는 GPU보다 **호스트 메모리와 object serialization**이 먼저 무너질 수 있습니다.

---

## 2.3 `/dev/shm` / Ray object store / shared memory 이슈

### 대표 증상
- Ray object store가 `/dev/shm`에 잡힘
- shared memory object open 실패
- SIGBUS 또는 object transfer 비정상
- 컨테이너에서는 재현되는데 bare-metal에선 덜 재현됨

### 공식 근거에서 보이는 원인
공식 설치 문서는 Docker 실행 예시에 **`--shm-size="10g"`** 를 사용합니다.  
또한 공식 이슈 로그에는 Ray가 **Plasma object store를 `/dev/shm`에 올리는 모습**이 보이며, 별도 이슈에서는 `unable to open shared memory object ... Too many open files`가 나옵니다.

### 확인 포인트
```bash
df -h /dev/shm
mount | grep shm
ls -lh /tmp/ray/session_latest/logs | head
```

### 조치 우선순위
1. 컨테이너에서라면 `/dev/shm` 크기를 확인하고, **기본값보다 충분히 크게** 잡습니다.  
   - 공식 예시는 10GB
   - 실제 대형 멀티트레이닝에서는 이보다 더 크게 잡아야 할 수 있습니다.
2. 한 번에 object store로 오가는 payload를 줄입니다.
   - 긴 response
   - 큰 `n`
   - 큰 이미지/멀티모달 payload
3. FD 부족과 함께 나타나면 **`/dev/shm` 문제와 FD 문제를 같이** 봅니다.
4. 문제가 SHM에서만 재현되면:
   - 컨테이너 런타임 설정
   - cgroup/shared memory mount
   - Ray object store 로그
   를 같이 확인합니다.

### 실무 팁
- `/dev/shm` 이슈는 종종 **OOM처럼 보이지 않고** SIGBUS/EOF/shared memory object open 실패로 나타납니다.
- “공식 Docker 예시가 10g를 쓴다”는 건, 적어도 **컨테이너 기본 SHM로는 부족할 수 있다는 신호**로 봐야 합니다.

---

## 2.4 GPU 메모리 OOM / rollout KV cache / actor-rollout weight sync OOM

### 대표 증상
- `CUDA out of memory`
- checkpoint resume 직후 OOM
- rollout 시작 시 메모리 폭증
- actor↔rollout 동기화에서 메모리 피크 급상승

### 공식 근거에서 보이는 원인
공식 config/perf 문서와 FSDP 확장 문서는 다음을 분명히 말합니다.

- `gpu_memory_utilization`은 rollout(vLLM/SGLang)의 GPU 메모리 점유 비율 제어에 중요
- 성능/메모리 튜닝의 핵심 knob는 **`*micro_batch_size_per_gpu` 또는 `use_dynamic_bsz=True` + `*_max_token_len_per_gpu`**
- **gradient checkpointing**, **activation offload**, **param/optimizer offload**는 메모리 절감을 위한 공식 옵션
- FSDP에서 `hf_weight_loader`는 **FULL_STATE_DICT를 모아오므로 peak memory가 더 큼**
- 공식 문서는 FSDP backend에서 **`dtensor_weight_loader`를 권장**합니다

### 즉시 줄일 설정
- `actor_rollout_ref.rollout.gpu_memory_utilization`
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`
- `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`
- `critic.ppo_micro_batch_size_per_gpu`
- `data.max_prompt_length`
- `data.max_response_length`
- `actor_rollout_ref.rollout.n`
- `actor_rollout_ref.rollout.max_num_batched_tokens`
- `actor_rollout_ref.rollout.max_num_seqs`

### 즉시 켤 설정
- `actor_rollout_ref.model.enable_gradient_checkpointing=True`
- `critic.model.enable_gradient_checkpointing=True`
- 필요 시 `actor_rollout_ref.model.enable_activation_offload=True`
- 필요 시 `critic.model.enable_activation_offload=True`
- 필요 시 `actor_rollout_ref.actor.fsdp_config.param_offload=True`
- 필요 시 `actor_rollout_ref.actor.fsdp_config.optimizer_offload=True`

### `use_dynamic_bsz=True`를 써야 하는 경우
길이 분산이 큰 데이터(짧은 샘플과 긴 샘플이 섞인 RL 데이터)에서는 공식 성능 튜닝 문서가 `use_dynamic_bsz=True`를 권장합니다.  
이 경우는 `*micro_batch_size_per_gpu` 대신 아래를 튜닝합니다.

- `actor_rollout_ref.actor.ppo_max_token_len_per_gpu`
- `critic.ppo_max_token_len_per_gpu`
- `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu`

### weight loader 관련 주의
- **대형 모델 + FSDP**라면 `hf` loader보다 `dtensor` 경로를 먼저 고려하세요.
- config 예시의 `dummy_dtensor`는 rollout 초기화 표현이고, 문서 설명대로 hybrid engine이 actor/rollout sync 시 대응 weight loader(`dtensor`)를 선택합니다.

### 실무 팁
메모리 문제를 잡을 때는 보통 아래 순서가 효율적입니다.

1. 길이(`max_prompt_length`, `max_response_length`) 축소  
2. `n` 축소  
3. `ppo_micro_batch_size_per_gpu` / `log_prob_micro_batch_size_per_gpu` 축소  
4. `gpu_memory_utilization` 하향  
5. gradient checkpointing / activation offload / param offload  
6. `hf` 경로 대신 `dtensor` 계열 로더 검토

---

## 2.5 GPU utilization 0% / 느린 학습 / hang처럼 보이는 stall

### 대표 증상
- GPU가 오랫동안 0%
- rollout은 도는 것 같지만 step이 거의 안 나감
- 멀티턴이나 async 모드에서 “한참 있다가 timeout”
- inference memory가 안 풀리는 것처럼 보임

### 공식 근거에서 보이는 원인
공식 오픈 이슈를 보면:

- **SGLang 멀티턴**에서 rollout worker가 search API를 치고도 **오래 걸린 뒤 끝나지 않는 문제**가 보고됨
- **agent_loop + sglang + async**에서 GPU utilization이 장시간 0%이고 generate가 비정상적으로 느린 문제 보고
- **fully async 공식 문서**는 현재 지원 모드를 **`megatron/fsdp + vllm`** 로 한정
- 공식 설치 문서는 **SGLang rollout이 “extensive development” 상태**라고 설명

즉, **SGLang async / multi-turn 조합은 아직 공식적으로 가장 안정적인 경로라고 보기 어렵습니다.**

### 확인 포인트
```bash
nvidia-smi dmon -s pucvmet
ray status
grep -R "timeout\|register center\|actor_names\|generate_sequences" /tmp/ray/session_latest/logs -n | tail -200
```

### 조치 우선순위
1. **async / multi-turn 문제는 우선 vLLM backend로 재현을 비교**합니다.
2. SGLang에서 hang가 보이면:
   - `n`
   - `max_prompt_length`
   - `max_response_length`
   - 멀티턴 turn 수
   - micro batch
   를 먼저 줄입니다.
3. fully async가 필요하면 문서 기준으로 **vLLM 경로를 우선** 검토합니다.
4. config 이름 변경/폐기가 있었던 케이스가 실제 이슈에 있으므로, **현재 문서/예제와 필드명이 맞는지** 확인합니다.
5. 성능 분석이 필요하면:
   - `ray_init.timeline_json_file=/tmp/ray_timeline.json`
   - Perfetto / `chrome://tracing`
   로 병목을 봅니다.

### 실무 팁
- “GPU 0%”는 항상 GPU 문제는 아닙니다.  
  실제로는 **actor 등록 대기, async server init, long-tail sample, search/tool I/O, Ray serialization**이 병목인 경우가 많습니다.

---

## 2.6 파일 디스크립터(FD) 부족 / `Too many open files`

### 대표 증상
- `RuntimeError: unable to open shared memory object ... Too many open files (24)`
- VLM/멀티모달 학습에서만 유독 빨리 재현
- `ulimit`은 높거나 unlimited인데도 발생

### 공식 근거에서 보이는 원인
공식 이슈 #2047은 `run_qwen2_5_vl-7b.sh` 실행 중 **shared memory object open 단계에서 FD 부족**이 나는 사례를 보여줍니다.  
즉, 단순 유저 셸의 `ulimit -n`만으로 설명되지 않고, **프로세스/서비스/SHM/멀티프로세싱이 합쳐진 FD 압력**일 수 있습니다.

### 확인 포인트
```bash
ulimit -n
cat /proc/sys/fs/file-max
lsof -p <PID> | wc -l
systemctl show --property=LimitNOFILE
```

### 조치 우선순위
1. 셸 `ulimit -n` 뿐 아니라 **systemd/service/container의 FD limit**도 확인합니다.
2. 멀티모달/대형 batch에서 open FD가 급증하면:
   - worker 수
   - batch size
   - long sequence
   - shared memory object 생성 빈도
   를 낮춥니다.
3. `/dev/shm` 문제와 함께 나타나면 SHM도 같이 키웁니다.
4. 재현 스크립트를 최소화하여 **텍스트-only와 멀티모달을 비교**합니다.

---

## 2.7 네트워크 / NCCL / 멀티노드 접속 실패

### 대표 증상
- `socketStartConnect: Connect to 169.254.0.1<...> failed`
- `Software caused connection abort`
- NCCL watchdog timeout
- 멀티노드에서만 hang 또는 connect abort

### 공식 근거에서 보이는 원인
공식 이슈 #726은 멀티노드 학습에서 **169.254.x.x 같은 link-local 주소**로 접속을 시도하다 실패하는 케이스를 보여줍니다.  
또한 공식 AMD 멀티노드 예시는 `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, `NCCL_CROSS_NIC`, `CUDA_DEVICE_MAX_CONNECTIONS` 등 **NIC/IB 관련 변수를 매우 명시적으로 잡고 있습니다.**

### 확인 포인트
```bash
ray status
hostname -I
ip addr
ibdev2netdev   # IB 환경이면
env | egrep "NCCL|CUDA_DEVICE_MAX_CONNECTIONS|RAY"
```

### 조치 우선순위
1. head node / worker node의 Ray address가 **실제 routable IP**인지 확인합니다.
2. 169.254.x.x 같은 주소가 보이면 NIC 선택이 잘못되었을 가능성이 큽니다.
3. IB/RoCE 환경이면 공식 예제처럼 **올바른 HCA / GID / cross-NIC 설정**을 확인합니다.
4. 방화벽, 포트, container network mode(`--net=host` 등), hostname resolution을 점검합니다.
5. 먼저 **1 node / 2 GPU**에서 성공시킨 뒤, 노드 수만 늘려 재현합니다.

### 실무 팁
- NCCL timeout은 “통신이 느리다”가 아니라, **통신 그룹 일부가 애초에 잘못된 NIC/IP로 묶였다**는 뜻인 경우가 많습니다.
- 멀티노드에서만 죽는다면 가장 먼저 **IP/NIC/port/firewall**을 의심하세요.

---

## 2.8 체크포인트 저장/로드 I/O 병목

### 대표 증상
- 저장이 너무 느림
- `CheckpointingException`
- resume는 되는데 다른 world size에서 깨짐
- 저장은 되지만 다음 step에서 OOM 또는 crash

### 공식 근거에서 보이는 원인
공식 이슈들에는 다음 계열이 반복됩니다.

- Megatron backend에서 checkpoint save가 **매우 느림**
- 특정 모델/조합에서 save 자체가 **CheckpointingException**으로 실패
- 멀티노드 `resume_from_path` 시 **world size가 바뀌면 FileNotFoundError**
- resume 직후 OOM 재현

### 조치 우선순위
1. **처음부터 save/resume를 작은 step 주기로 미리 검증**합니다.
2. 가능하면 **빠른 로컬 SSD**에 먼저 저장하고, 이후 외부 스토리지로 옮깁니다.
3. 재시작 시에는 **world size를 동일하게 유지**합니다.
4. `resume_mode`, `resume_from_path`, `default_local_dir`를 명시적으로 관리합니다.
5. 대형 모델에서는 checkpoint save 주기를 너무 촘촘하게 두지 않습니다.

---

## 3. 설정 이슈(파라미터)

## 3.1 `trainer.nnodes`, `n_gpus_per_node`, TP/SP/world size 불일치

### 흔한 실수
- `trainer.nnodes`가 실제 노드 수와 다름
- `trainer.n_gpus_per_node`가 실제 CUDA_VISIBLE_DEVICES와 다름
- `actor_rollout_ref.rollout.tensor_model_parallel_size`가 rollout GPU 수와 맞지 않음
- async/multi-turn에서 actor 수 기대값과 실제 등록 actor 수가 다름

### 대표 증상
- actor registration timeout
- `instance_id ... has 0 actors, but ... is expected`
- NCCL group init 실패
- rollout backend만 따로 뜨다 assert

### 대응
- 먼저 **single node / TP=1**로 최소 재현합니다.
- 그 다음 순서대로 늘립니다.
  1. node 1개
  2. TP 1
  3. 작은 batch
  4. multi-turn off
  5. async off
  6. 그 후 하나씩 확장

---

## 3.2 batch size 계열 파라미터 혼동

공식 성능 문서는 **`*micro_batch_size_per_gpu`를 쓰고, 구식 `*micro_batch_size`는 피하라**고 안내합니다.  
또한 mini-batch는 global 개념이고, micro-batch는 local(per-GPU) 개념입니다.

### 기억할 원칙
- **`ppo_mini_batch_size`는 global**
- **`ppo_micro_batch_size_per_gpu`는 local**
- dynamic batch size를 쓰면 micro-batch보다 **token cap**을 튜닝

### 자주 나는 실수
- `ppo_mini_batch_size`만 키우고 micro-batch를 같이 키워 OOM
- `use_dynamic_bsz=True`인데 여전히 micro-batch를 집요하게 튜닝
- actor/ref/critic/log_prob micro-batch를 전부 같은 값으로 둠

### 추천 접근
- actor가 가장 메모리 빡빡하므로 **actor 기준으로 먼저 맞춤**
- ref/log_prob/critic은 공식 문서처럼 **더 크게 잡을 수 있음**
- 긴 sequence가 많으면 `use_dynamic_bsz=True` 고려

---

## 3.3 길이 관련 설정: `max_prompt_length`, `max_response_length`, `truncation`

공식 config 설명상:

- `data.truncation` 기본은 `error`
- prompt가 길면 에러를 던짐
- 필요하면 `left`, `right`, `middle` 사용 가능

### 대표 증상
- 시작 직후 overlong prompt 관련 에러
- 멀티턴에서 뒤로 갈수록 길이가 불어나 갑자기 OOM
- `n>1` + 긴 response에서 rollout KV cache 폭증

### 대응 원칙
1. 에러를 피하려고 무조건 `max_prompt_length`를 키우기보다  
   **먼저 데이터 특성을 보세요.**
2. reasoning / multi-turn이면 `max_response_length`가 메모리에 더 치명적일 때가 많습니다.
3. `truncation=middle`은 “긴 문서의 앞/뒤는 살리고 중간을 버리는” 용도로 실전성이 있습니다.
4. 멀티턴 RL은 turn이 쌓이면서 길이가 커지므로, **초기 dry run에서는 turn 수도 같이 줄이세요.**

---

## 3.4 rollout backend 선택: vLLM vs SGLang vs fully async

### 공식 문서 기준
- vLLM은 **0.8.3+가 안정성 테스트 대상**
- 성능을 위해 **`VLLM_USE_V1=1` 권장**
- SGLang rollout은 **extensive development**
- fully async는 **현재 `megatron/fsdp + vllm` 지원**

### 실전 해석
- **안정성이 최우선이면 vLLM 우선**
- **SGLang async / multi-turn은 사전 검증 비용을 높게 잡아야 함**
- fully async를 꼭 써야 한다면 현재 문서 기준으로는 **SGLang보다 vLLM 쪽이 문서화가 명확**

---

## 3.5 weight loader / FSDP 관련 파라미터

### 핵심 포인트
- `hf` loader: FULL_STATE_DICT 기반 → **peak memory 큼**
- `dtensor` loader: layer-by-layer / sharded path → **대형 모델에서 유리**
- FSDP 메모리 옵션:
  - `param_offload`
  - `optimizer_offload`
  - `enable_activation_offload`
  - `enable_gradient_checkpointing`

### 대표 실수
- 대형 모델인데 `hf` loader 계열을 써서 sync 시 OOM
- checkpointing/offload를 다 끄고 micro-batch만 무식하게 키움
- actor와 rollout dtype 정렬이 안 됨

### 권장
- **대형 FSDP 모델은 dtensor 우선**
- 메모리 부족 시:
  1. gradient checkpointing
  2. activation offload
  3. param/optimizer offload
  순으로 고려

---

## 3.6 inference/training precision mismatch

공식 FAQ는 `actor/grad_norm`이 계속 커질 때 **inference engine과 training 간 precision mismatch** 가능성을 설명합니다.

### 공식 진단법
```yaml
actor_rollout_ref.rollout.calculate_log_probs=True
```

이후 생기는 지표:
- `training/rollout_probs_diff_mean`

### 공식 해석 기준
- 보통 **0.005 이하** 기대
- **0.01 초과**면 precision 문제 의심

### 공식 workaround
아래 조건이 겹치면
- non-Hopper 계열 GPU
- vLLM의 알려진 이슈 영향 버전/경로
- 긴 input/output, 특히 multi-turn reasoning

다음을 추가:
```yaml
+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=True
```

### 실무 팁
이 문제는 “학습은 되는데 결과가 점점 이상해지는” 형태로 보이기 쉽습니다.  
그래서 **OOM이나 crash보다 더 늦게 발견**됩니다.

---

## 3.7 Triton / compile / fused kernel 관련 설정

### 대표 증상
- Triton `compile_module_from_src` error
- fused kernel 계열에서 초기 컴파일 실패
- traceback이 길고 원인이 모호함

### 공식 대응
FAQ는 이 경우 **`use_torch_compile`을 꺼서 JIT compilation을 비활성화**하라고 안내합니다.

### 추천 순서
1. 먼저 `use_torch_compile=False`
2. 여전히 문제면 attention implementation을 `eager`로 낮춰 재현
3. fused kernel / custom kernel / extension을 하나씩 다시 켬

---

## 3.8 checkpoint / resume 관련 파라미터

### 꼭 기억할 것
- `trainer.resume_mode`: `disable`, `auto`, `resume_path`
- `trainer.resume_from_path`
- `trainer.default_local_dir`
- `trainer.save_freq`
- `trainer.remove_previous_ckpt_in_save`
- `trainer.del_local_ckpt_after_load`

### 자주 나는 실수
- checkpoint를 저장하긴 했는데 save path를 확실히 모름
- world size가 바뀐 상태로 resume 시도
- 대형 모델에서 save_freq를 너무 촘촘히 둠
- “장애 대비”를 위해 resume를 켰지만, 정작 첫 checkpoint 복구는 테스트하지 않음

### 권장 원칙
- **첫 checkpoint 저장 후 즉시 복구 테스트**
- 멀티노드 재개 시 **world size를 바꾸지 않는 것**을 기본 원칙으로
- 대형 모델은 save I/O를 학습 시간 예산에 포함

---

## 4. 자주 보이는 에러메시지와 1차 대응

| 에러/증상 | 보통 의심할 원인 | 1차 대응 | 추가 확인 |
|---|---|---|---|
| `Failed to register worker to Raylet` | SLURM CPU 제약, Ray CPU 인식 과다, agent/heartbeat 문제 | `ray_init.num_cpus` 명시 | heartbeat, grpcio, OOM, agent 로그 |
| `The node has been marked dead because the detector missed too many heartbeats` | Ray agent 실패, CPU starvation, 포트 충돌, OOM | CPU 여유 확보, agent 로그 확인 | `grpcio`, system logs |
| `socketStartConnect: Connect to 169.254.x.x... failed` | 잘못된 NIC/IP 선택, 멀티노드 네트워크 설정 오류 | routable IP/NIC 재확인 | IB/NCCL env, firewall |
| `CUDA out of memory` | 길이/배치/`n`/KV cache/weight sync 피크 | 길이·micro-batch·`n` 축소 | `gpu_memory_utilization`, dtensor loader |
| `CUDA error: an illegal memory access was encountered` | rollout 엔진/커널/버전 조합 문제 | vLLM 해당 버전 문서 확인, eager 경로 시도 | `CUDA_LAUNCH_BLOCKING=1`, kernel 옵션 |
| `RuntimeError: CUDA error: misaligned address` | 비동기 커널 에러, NCCL watchdog 연쇄 | blocking 디버깅, batch/길이 축소 | fused kernel/attention implementation |
| `instance_id ... has 0 actors, but ... is expected` | async server actor 등록 불일치, world size/TP mismatch | 최소 구성(TP=1, single node)으로 재현 | actor registration/logs |
| rollout이 오래 걸리다 끝나지 않음 | async multi-turn/SGLang hang 계열 | vLLM로 비교 재현 | config drift, tool/search I/O |
| GPU util이 오랫동안 0% | GPU보다 Ray/async init/I/O가 병목 | Ray timeline 수집 | actor 생성, generate 단계 지연 |
| `unable to open shared memory object ... Too many open files (24)` | FD 부족 + SHM 객체 폭증 | FD limit, SHM, worker 수 점검 | systemd/container limit |
| `Ray SIGBUS error on my dataset` | DRAM 또는 object store/serialization 문제 | 데이터 축소, shard화 | OOM killer, SHM, dataset payload |
| `CheckpointingException` | 대형 모델 save 실패 / 경로 / backend 이슈 | save 경로·주기·local SSD 확인 | same world size resume 여부 |

---

## 5. 공식 GitHub에 제기된 문제들(멀티트레이닝 관점 선별)

> 아래는 “모든 이슈”가 아니라, 멀티노드/멀티GPU/Ray/rollout/backend/체크포인트 관점에서 **재발성이 높은 이슈**를 추린 목록입니다.

| 이슈 | 상태(2026-03-24 기준) | 분류 | 증상 요약 | 실무 해석 |
|---|---|---|---|---|
| `#523` Failed to register worker to Raylet on slurm cluster | Open | Ray/CPU | SLURM 환경에서 raylet 등록 실패 | CPU/SLURM 제약과 Ray 자원 인식부터 봐야 함 |
| `#726` multi-node training nccl issue...169.254.0.1 | Closed | NCCL/Network | link-local 주소로 connect abort | NIC/IP 선택 오류의 대표 사례 |
| `#1365` Multi-Turn Rollout on a single GPU | Closed | Async rollout | `vllm_async_server.py` assertion | actor 등록/async init 경로 불일치 |
| `#1632` Raylet crashes during PPO Quickstart on A800 | Open | Ray/Heartbeat | raylet 등록 실패 후 heartbeat/agent 문제 | 단순 raylet 문제가 아니라 시스템/agent 문제일 수 있음 |
| `#2001` megatron + vllm async: `execute_method` 없음 | Open | Backend 조합 | Megatron + vLLM async에서 attribute error | backend 조합/experimental path 검증 필요 |
| `#2047` Too many open files | Closed | FD/SHM | shared memory object open 실패 | 멀티모달 + multiprocessing에서 FD 압력 점검 필요 |
| `#2265` multi-node training `resume_from_path` | Closed | Resume | 다른 world size로 resume 시 FileNotFound | resume는 같은 world size 유지가 안전 |
| `#2445` sglang multiturn problem: rollout never finishes | Open | SGLang multi-turn | search API 후 오래 걸리다 끝나지 않음 | SGLang 멀티턴 안정성 사전 검증 필요 |
| `#3114` GPU utilization stays at 0% for a long time | Open | Async/SGLang 성능 | generate가 지나치게 느리고 GPU util 0% | GPU가 아니라 async/Ray/I/O 병목일 수 있음 |
| `#3218` Ray SIGBUS error on my dataset | Open | DRAM/object store | dataset 처리 중 SIGBUS, actor death | 데이터 크기/직렬화/호스트 메모리 우선 점검 |
| `#3579` CUDA error: misaligned address | Open | CUDA/NCCL | watchdog + misaligned address | 버전/커널/배치/길이 축소 후 blocking 디버깅 |
| `#4990` Qwen3 VL 30B MoE save checkpoint fail on megatron | Closed | Checkpoint | save 단계 CheckpointingException | Megatron 대형 모델 save 경로는 사전 검증 필수 |
| `#5119` vLLM 0.14.0 + verl(main) step 4x slower + NCCL timeouts | Open | 버전/성능 | 최신 조합에서 step time 급증과 timeout | 새 버전 도입 시 반드시 pin & benchmark 필요 |
| `#5474` Is SGLang supported as rollout backend in fully async mode? | Open | Async backend 지원 범위 | 문서/스크립트 간 지원 범위 혼선 | fully async는 현재 문서상 vLLM 우선 해석이 안전 |

---

## 6. 증상별 플레이북

## 6.1 시작도 못 하고 바로 죽는다

### 가장 먼저 의심
1. `ray_init.num_cpus` 미설정
2. `trainer.nnodes` / `n_gpus_per_node` / TP 불일치
3. `/dev/shm` 너무 작음
4. 잘못된 NIC/IP
5. backend 조합이 현재 문서 지원 범위 밖

### 권장 순서
1. single node / 1~2 GPU / TP=1 / multi-turn off / async off
2. `ray status`
3. `free -h`, `df -h /dev/shm`
4. `nvidia-smi`
5. raylet / agent / NCCL 로그 확인
6. 그 다음에만 멀티노드로 확대

---

## 6.2 몇 step 돌다가 죽는다

### 가장 먼저 의심
1. rollout KV cache 축적
2. 길이/`n` 증가에 따른 token 폭증
3. actor↔rollout sync 피크 메모리
4. checkpoint save 시점의 추가 메모리/I/O 부하
5. precision mismatch로 grad_norm 폭주

### 즉시 할 일
- `n` 절반
- `max_response_length` 절반
- `ppo_micro_batch_size_per_gpu` 절반
- `log_prob_micro_batch_size_per_gpu` 절반
- `gpu_memory_utilization` 하향
- `calculate_log_probs=True`로 mismatch 확인
- checkpoint 직전/직후 메모리 패턴 확인

---

## 6.3 느리지만 죽지는 않는다

### 가장 먼저 의심
1. GPU가 아니라 Ray/I/O/tool/search 병목
2. SGLang async/multi-turn 병목
3. batch size가 너무 작아 launch overhead 우세
4. dynamic batch size 미사용
5. remove padding 미사용
6. save/test 주기 과도

### 조치
- `use_remove_padding=True`
- 길이 분산 크면 `use_dynamic_bsz=True`
- `ray_init.timeline_json_file`로 타임라인 수집
- validation / checkpoint 주기 완화
- fully async가 필요하면 vLLM 경로 우선 검토

---

## 6.4 resume / checkpoint가 유독 불안정하다

### 가장 먼저 의심
1. world size 변경
2. save path / local vs remote path 혼동
3. Megatron save 자체 병목
4. save 주기 과밀
5. resume는 되었지만 바로 다음 step에서 메모리 피크

### 권장
- 첫 checkpoint에서 **즉시 resume 테스트**
- same world size 유지
- local SSD 우선 저장
- save_freq 완화
- 대형 모델이면 save 자체를 성능 예산에 포함

---

## 7. 보수적으로 시작하는 예시 설정(실전 제안)

> 아래는 **“먼저 성공시키기 위한 보수적 시작점”** 입니다. 공식 default 자체가 아니라, 위 공식 문서/이슈 패턴을 바탕으로 한 안정성 우선 예시입니다.

```yaml
# Ray / debug
ray_init:
  num_cpus: 24
  timeline_json_file: /tmp/ray_timeline.json

data:
  max_prompt_length: 1024
  max_response_length: 512
  filter_overlong_prompts: True
  truncation: error

actor_rollout_ref:
  model:
    use_remove_padding: True
    enable_gradient_checkpointing: True
    enable_activation_offload: False

  actor:
    ppo_mini_batch_size: 64
    ppo_micro_batch_size_per_gpu: 1
    use_dynamic_bsz: False

  rollout:
    name: vllm
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.35
    log_prob_micro_batch_size_per_gpu: 1
    free_cache_engine: True
    enforce_eager: True
    load_format: dummy_dtensor
    n: 1

  ref:
    log_prob_micro_batch_size_per_gpu: 1

trainer:
  nnodes: 1
  n_gpus_per_node: 2
  save_freq: 20
  test_freq: 20
  resume_mode: disable
```

### 이 설정으로 먼저 보는 것
- **정상 기동**
- **1 step 이상 안정 진행**
- **checkpoint 1회 저장**
- **resume 1회 확인**
- 이후에만
  - node 수 증가
  - GPU 수 증가
  - TP 증가
  - `n` 증가
  - multi-turn on
  - async on
  순으로 확장합니다.

---

## 8. 진단용 명령어 모음

### 노드/자원 상태
```bash
hostname
hostname -I
nvidia-smi
free -h
df -h /dev/shm
ulimit -n
ray status
```

### Ray 로그
```bash
ls -lah /tmp/ray/session_latest/logs | head -50
grep -R "raylet\|heartbeat\|register worker\|SIGBUS\|OOM\|NCCL\|actor_names" /tmp/ray/session_latest/logs -n | tail -200
```

### 시스템 로그
```bash
dmesg | egrep -i "oom|killed process|sigbus|nv|nvidia"
journalctl -k -n 200
```

### 열린 FD 수
```bash
lsof -p <PID> | wc -l
cat /proc/sys/fs/file-max
```

### 네트워크 / NIC
```bash
ip addr
env | egrep "NCCL|CUDA_DEVICE_MAX_CONNECTIONS|RAY"
ibdev2netdev   # IB 환경
```

### 프로파일링
```bash
# config
ray_init.timeline_json_file=/tmp/ray_timeline.json
# 종료 후 Perfetto 또는 chrome://tracing 에서 확인
```

---

## 9. 최종 권장 운영 원칙

1. **single-node 최소 구성을 먼저 성공**시키고 확장하세요.  
2. 멀티노드 문제는 대부분 **네트워크/NIC/IP + Ray CPU/agent + SHM** 중 하나에서 시작합니다.  
3. OOM은 단순히 GPU만 보지 말고 **DRAM / SHM / serialization**도 같이 보세요.  
4. 대형 FSDP 모델은 **`hf`보다 `dtensor` 계열**을 먼저 고려하세요.  
5. SGLang async/multi-turn은 문서와 이슈를 보면 아직 검증 비용이 큽니다. **안정성 우선이면 vLLM부터** 보세요.  
6. checkpoint/resume는 “나중에 필요할 때”가 아니라 **처음부터 테스트**해야 합니다.  
7. 새 버전(vLLM/Ray/Torch)을 올릴 때는 **pin + benchmark + rollback path**를 반드시 준비하세요.

---

## 10. 참고 근거(공식 문서 / 공식 이슈)

### 공식 저장소 / 문서
- [D1] 공식 저장소 README — <https://github.com/verl-project/verl>
- [D2] 공식 문서 홈 — <https://verl.readthedocs.io/en/latest/>
- [D3] Multinode Training — <https://verl.readthedocs.io/en/latest/start/multinode.html>
- [D4] Config Explanation — <https://verl.readthedocs.io/en/latest/examples/config.html>
- [D5] Performance Tuning Guide — <https://verl.readthedocs.io/en/latest/perf/perf_tuning.html>
- [D6] FAQ — <https://verl.readthedocs.io/en/latest/faq/faq.html>
- [D7] Installation — <https://verl.readthedocs.io/en/latest/start/install.html>
- [D8] Add models with the FSDP backend — <https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html>
- [D9] Hardware Resource Needed for RL — <https://verl.readthedocs.io/en/latest/perf/device_tuning.html>
- [D10] Recipe: Fully Async Policy Trainer — <https://verl.readthedocs.io/en/latest/advance/fully_async.html>

### 공식 GitHub 이슈
- [I523] Failed to register worker to Raylet on slurm cluster — <https://github.com/verl-project/verl/issues/523>
- [I726] multi-node training nccl issue. socketStartConnect... — <https://github.com/verl-project/verl/issues/726>
- [I1365] Multi-Turn Rollout on a single GPU — <https://github.com/verl-project/verl/issues/1365>
- [I1632] Raylet crashes during PPO Quickstart on A800 GPUs — <https://github.com/verl-project/verl/issues/1632>
- [I2001] megatron + vllm async: AttributeError ... execute_method — <https://github.com/verl-project/verl/issues/2001>
- [I2047] Too many open files — <https://github.com/verl-project/verl/issues/2047>
- [I2265] multi-node training resume_from_path — <https://github.com/verl-project/verl/issues/2265>
- [I2445] sglang multiturn problem: rollout never finishes — <https://github.com/verl-project/verl/issues/2445>
- [I3114] GPU utilization stays at 0% for a long time — <https://github.com/verl-project/verl/issues/3114>
- [I3218] Ray SIGBUS error on my dataset — <https://github.com/verl-project/verl/issues/3218>
- [I3579] RuntimeError: CUDA error: misaligned address — <https://github.com/verl-project/verl/issues/3579>
- [I4990] Qwen 3 VL 30b moe training fails when save checkpoint on megatron — <https://github.com/verl-project/verl/issues/4990>
- [I5119] vLLM 0.14.0 + verl(main) step time becomes ~4× slower, with occasional NCCL timeouts — <https://github.com/verl-project/verl/issues/5119>
- [I5474] Is SGLang supported as a Rollout Backend in Fully Async Mode — <https://github.com/verl-project/verl/issues/5474>

---

## 11. 문서 작성 시 핵심적으로 반영한 공식 포인트 요약

- 공식 FAQ: `Failed to register worker with raylet`는 **SLURM CPU 제약**과 관련될 수 있으며 `ray_init.num_cpus` 설정을 권장
- 공식 config 문서: dataset parquet는 **메모리로 모두 읽으므로 너무 크면 안 됨**
- 공식 config/성능 문서: `*micro_batch_size_per_gpu`, `use_dynamic_bsz`, `ppo_max_token_len_per_gpu`, `use_remove_padding`, `gradient_checkpointing`, `activation_offload`가 핵심 튜닝 포인트
- 공식 FSDP 확장 문서: `hf_weight_loader`는 peak memory가 더 크고, **`dtensor_weight_loader` 권장**
- 공식 FAQ: `training/rollout_probs_diff_mean`으로 inference/training mismatch 진단 가능, 필요 시 `disable_cascade_attn=True`
- 공식 설치/async 문서: **vLLM이 상대적으로 명시적인 주 경로**, fully async는 현재 `megatron/fsdp + vllm` 지원
- 공식 이슈들: Raylet 등록 실패, NCCL NIC 문제, SGLang 멀티턴 hang, FD 부족, SIGBUS, checkpoint save/resume 이슈가 반복적으로 등장
