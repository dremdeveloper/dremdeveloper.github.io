# LLaMA-Factory 가이드

LLaMA-Factory는 대규모 언어 모델과 비전-언어 모델의 학습, 미세조정, 추론, 평가, 내보내기를 하나의 CLI와 WebUI로 묶어 제공하는 통합 도구다. 처음에는 기능을 넓게 잡기보다 `SFT + LoRA + 텍스트 데이터셋` 경로로 한 번 끝까지 통과하는 편이 가장 안정적이다.

LLaMA-Factory를 처음 붙일 때는 다음 순서로 진행한다.

1. 설치를 완료한다.
2. 공식 예제가 실행되는지 먼저 확인한다.
3. 커스텀 데이터셋을 `data/dataset_info.json`에 등록한다.
4. 예제 YAML을 복사해 최소 항목만 수정한다.
5. 학습 후 `chat`, `webchat`, `api`, `export` 순서로 확인한다.

---

## 1. 빠르게 시작하기

환경이 준비되어 있다면 공식 Quickstart부터 실행한다.

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
llamafactory-cli chat examples/inference/qwen3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
```

각 명령의 의미는 다음과 같다.

- `train`: LoRA 기반 SFT 학습
- `chat`: 학습된 어댑터를 불러와 대화형 추론
- `export`: 베이스 모델과 LoRA 어댑터를 병합해 완성 모델로 내보내기

처음 1회는 이 흐름이 정상 동작하는지 확인한 뒤, 커스텀 데이터셋과 커스텀 설정으로 넘어간다.

---

## 2. 핵심 개념

### `model_name_or_path`
학습 또는 추론에 사용할 베이스 모델이다. Hugging Face 모델 ID나 로컬 경로를 사용한다.

### `template`
모델이 기대하는 대화 포맷이다. Instruct/Chat 모델은 반드시 대응하는 템플릿을 사용한다. 학습과 추론에서 템플릿은 반드시 같아야 한다.

예시:

- Llama 3 Instruct 계열: `llama3`
- Qwen3 계열: `qwen3`
- Qwen3 비추론 템플릿: `qwen3_nothink`

### `stage`
학습 단계를 지정한다.

- `pt`: pre-training
- `sft`: supervised fine-tuning
- `rm`: reward modeling
- `ppo`, `dpo`, `kto`, `orpo`

처음에는 `sft`로 시작한다.

### `finetuning_type`
가중치를 어느 범위까지 업데이트할지 정한다.

- `lora`: 어댑터만 학습
- `freeze`: 일부 레이어만 학습
- `full`: 전체 가중치 학습

처음에는 `lora`를 사용한다.

### `dataset`
`dataset_info.json`에 등록한 데이터셋 이름이다. 여러 개를 사용할 때는 쉼표로 연결한다.

```yaml
dataset: my_sft
```

### `output_dir`
학습 결과를 저장하는 경로다. LoRA 학습에서는 이 경로가 이후 `adapter_name_or_path`로 다시 사용된다.

### `adapter_name_or_path`
LoRA 어댑터 경로다. 추론과 병합에서 사용한다. 일반적으로 학습 때 사용한 `output_dir`와 동일하다.

---

## 3. 설치

### 3.1 사전 점검

Linux에서는 먼저 다음을 확인한다.

```bash
uname -m && cat /etc/*release
gcc --version
nvcc -V
```

CUDA를 새로 설치할 경우 공식 문서는 CUDA 12.2를 권장한다.

### 3.2 기본 설치

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
pip install -r requirements/metrics.txt
```

환경 충돌이 있으면 다음 방식으로 설치를 시도한다.

```bash
pip install --no-deps -e .
```

### 3.3 설치 확인

```bash
llamafactory-cli version
```

버전 정보가 출력되면 기본 설치는 끝난다.

### 3.4 자주 같이 설치하는 의존성

필수는 아니지만 다음 기능을 쓸 때는 관련 패키지가 필요하다.

- `bitsandbytes`: QLoRA
- `vllm`: 고속 추론 및 API
- `deepspeed`: 대규모 분산 학습
- `awq`, `gptq`, `aqlm`, `hqq`, `eetq`: 양자화 모델

---

## 4. 작업 디렉터리

처음 프로젝트는 다음 구조로 두면 관리하기 쉽다.

```text
LLaMA-Factory/
├─ data/
│  ├─ dataset_info.json
│  └─ my_sft.json
├─ configs/
│  ├─ train_my_sft.yaml
│  ├─ infer_my_sft.yaml
│  └─ merge_my_sft.yaml
├─ examples/
├─ saves/
└─ src/
```

`data/`에는 데이터 파일과 `dataset_info.json`을 둔다. `configs/`에는 직접 사용하는 YAML을 모아 둔다. `saves/`는 학습 결과가 쌓이는 경로다.

---

## 5. 데이터 준비

처음에는 **Alpaca 형식**으로 시작하는 편이 가장 쉽다. 멀티턴 대화나 tool-calling이 필요할 때만 ShareGPT 형식으로 넘어간다.

### 5.1 Alpaca 형식

가장 단순한 SFT 데이터는 다음 구조를 사용한다.

- `instruction`: 필수
- `input`: 선택
- `output`: 필수
- `system`: 선택
- `history`: 선택

`data/my_sft.json`

```json
[
  {
    "instruction": "한 문장으로 자기소개를 해줘.",
    "input": "",
    "output": "안녕하세요. 저는 사용자의 요청을 돕는 AI 어시스턴트입니다."
  },
  {
    "instruction": "서울을 두 문장으로 설명해줘.",
    "input": "",
    "output": "서울은 대한민국의 수도이자 경제·문화 중심지입니다. 전통과 현대가 함께 보이는 대도시입니다."
  },
  {
    "instruction": "아래 문장을 비즈니스 메일 문체로 바꿔줘.",
    "input": "내일까지 자료 보내주세요.",
    "output": "안녕하세요. 검토를 위해 관련 자료를 내일까지 전달해 주시면 감사하겠습니다."
  }
]
```

LLaMA-Factory는 SFT에서 `instruction`과 `input`을 합쳐 최종 사용자 입력으로 사용하고, `output`을 정답으로 사용한다. `system`이 있으면 시스템 프롬프트로 사용한다. `history`를 넣으면 이전 대화까지 함께 학습한다.

### 5.2 Alpaca 멀티턴 예시

멀티턴을 빠르게 붙이고 싶다면 `history`를 사용한다.

```json
[
  {
    "instruction": "오늘 날씨가 어때?",
    "input": "",
    "output": "맑고 따뜻한 편입니다.",
    "history": [
      ["오늘 비 와?", "오늘은 비 소식이 없습니다."],
      ["밖에 나가기 괜찮아?", "네. 야외 활동하기 좋습니다."]
    ]
  }
]
```

### 5.3 `dataset_info.json` 등록

커스텀 데이터셋은 반드시 `data/dataset_info.json`에 등록한다. 파일이 없다면 생성하고, 이미 있으면 기존 JSON 객체에 항목을 추가한다.

```json
{
  "my_sft": {
    "file_name": "my_sft.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system",
      "history": "history"
    }
  }
}
```

여기서 `my_sft`가 YAML의 `dataset:` 값이 된다.

### 5.4 ShareGPT 형식

다음 조건 중 하나라도 있으면 ShareGPT 형식을 사용한다.

- 역할이 `user/assistant` 외에 `observation`, `function_call`, `tool`까지 필요함
- 대화 턴을 명시적으로 관리해야 함
- OpenAI 형식 `messages`를 그대로 가져오고 싶음

`data/my_sharegpt.json`

```json
[
  {
    "conversations": [
      {"from": "human", "value": "자기소개를 해줘."},
      {"from": "gpt", "value": "안녕하세요. 저는 사용자의 질문에 답하는 AI 어시스턴트입니다."},
      {"from": "human", "value": "한 줄로 줄여줘."},
      {"from": "gpt", "value": "안녕하세요. 질문에 답하는 AI입니다."}
    ],
    "system": "친절하고 간결하게 답변한다."
  }
]
```

이에 대응하는 등록은 다음과 같다.

```json
{
  "my_sharegpt": {
    "file_name": "my_sharegpt.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "system": "system"
    }
  }
}
```

### 5.5 OpenAI `messages` 형식

OpenAI 스타일 `messages`는 ShareGPT의 특수 케이스로 처리할 수 있다.

```json
[
  {
    "messages": [
      {"role": "system", "content": "친절하고 정확하게 답변한다."},
      {"role": "user", "content": "대한민국 수도는 어디야?"},
      {"role": "assistant", "content": "대한민국의 수도는 서울입니다."}
    ]
  }
]
```

등록은 다음과 같이 한다.

```json
{
  "my_openai": {
    "file_name": "my_openai.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

### 5.6 데이터 파일 점검 기준

학습 전에 다음 항목을 반드시 확인한다.

- JSON 문법 오류가 없는지
- 문자열 필드에 `null`이 들어가지 않았는지
- Alpaca 형식인데 `instruction`, `output`이 비어 있지 않은지
- ShareGPT 형식인데 `formatting: sharegpt`가 빠지지 않았는지
- `dataset_info.json`의 필드명이 실제 JSON 키와 정확히 같은지

가장 흔한 실패 원인은 **파일 내용과 `dataset_info.json`의 등록 정보가 서로 다르거나, 실제 데이터 형식과 `formatting` 선언이 맞지 않는 경우**다.

---

## 6. 첫 번째 학습 실행

처음에는 새 YAML을 처음부터 만들지 말고, 사용하는 모델과 가장 가까운 예제를 복사해서 수정한다.

예시:

```bash
cp examples/train_lora/qwen3_lora_sft.yaml configs/train_my_sft.yaml
```

그 다음 아래 항목만 먼저 바꾼다.

- `model_name_or_path`
- `dataset`
- `template`
- `output_dir`
- 배치 크기와 epoch

### 6.1 최소 학습 YAML

`configs/train_my_sft.yaml`

```yaml
### model
model_name_or_path: your_model_name_or_path

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: my_sft
template: your_template
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/my_model/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```

### 6.2 필수 항목만 이해하고 시작하기

- `stage: sft` — 지도 미세조정
- `finetuning_type: lora` — LoRA 방식 사용
- `lora_target: all` — LoRA를 적용할 모듈
- `dataset: my_sft` — `dataset_info.json`에 등록한 이름
- `template` — 모델과 일치하는 템플릿
- `cutoff_len` — 최대 길이
- `output_dir` — 어댑터 저장 경로

### 6.3 템플릿 선택 기준

템플릿은 반드시 모델과 맞아야 한다.

- Instruct/Chat 모델은 해당 모델 전용 템플릿을 사용한다.
- 학습과 추론에서 같은 템플릿을 사용한다.
- Qwen3 계열처럼 추론/비추론 버전이 나뉜 모델은 `qwen3`, `qwen3_nothink`처럼 서로 다른 템플릿을 사용한다.

모델별 템플릿은 README의 지원 모델 표와 `src/llamafactory/extras/constants.py`, `src/llamafactory/data/template.py`를 함께 확인하면 가장 빠르다.

### 6.4 학습 실행

```bash
llamafactory-cli train configs/train_my_sft.yaml
```

특정 GPU만 사용하려면 다음과 같이 실행한다.

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/train_my_sft.yaml
```

YAML의 일부 값을 즉석에서 바꾸고 싶다면 뒤에 덮어쓴다.

```bash
llamafactory-cli train configs/train_my_sft.yaml \
    learning_rate=1e-5 \
    logging_steps=1
```

### 6.5 첫 실행에서 권장하는 설정

처음에는 다음처럼 보수적으로 잡는 편이 좋다.

- `max_samples: 100` 또는 `1000`으로 시작
- `per_device_train_batch_size: 1`
- `gradient_accumulation_steps`로 유효 배치 크기 확보
- `num_train_epochs: 1` 또는 `3`
- 데이터가 아주 적으면 `val_size: 0.0`으로 시작

학습이 정상 종료되면 `output_dir` 아래에 LoRA 체크포인트가 저장된다. 이후 추론과 병합에서 이 경로를 그대로 사용한다.

---

## 7. 추론

LLaMA-Factory는 `chat`, `webchat`, `api`, 배치 추론을 지원한다. 기본 추론 엔진은 Hugging Face이며, `infer_backend: vllm`을 지정하면 vLLM을 사용할 수 있다.

### 7.1 원본 모델 추론

원본 모델만 시험할 때는 베이스 모델과 템플릿만 있으면 된다.

`configs/infer_base.yaml`

```yaml
model_name_or_path: your_model_name_or_path
template: your_template
infer_backend: huggingface
```

실행:

```bash
llamafactory-cli chat configs/infer_base.yaml
```

### 7.2 LoRA 모델 추론

LoRA 학습 결과를 사용할 때는 어댑터 경로와 미세조정 방식을 함께 넣는다.

`configs/infer_my_sft.yaml`

```yaml
model_name_or_path: your_model_name_or_path
adapter_name_or_path: saves/my_model/lora/sft
template: your_template
finetuning_type: lora
infer_backend: huggingface
```

실행:

```bash
llamafactory-cli chat configs/infer_my_sft.yaml
```

브라우저 채팅 인터페이스를 쓰려면 다음과 같이 실행한다.

```bash
llamafactory-cli webchat configs/infer_my_sft.yaml
```

### 7.3 API 서비스

OpenAI 스타일 API로 띄우려면 다음과 같이 실행한다.

```bash
API_PORT=8000 CUDA_VISIBLE_DEVICES=0 \
llamafactory-cli api configs/infer_my_sft.yaml
```

간단한 호출 예시는 다음과 같다.

```python
from openai import OpenAI

client = OpenAI(api_key="0", base_url="http://0.0.0.0:8000/v1")
messages = [{"role": "user", "content": "Who are you?"}]
result = client.chat.completions.create(
    messages=messages,
    model="your_model_name_or_path"
)
print(result.choices[0].message)
```

### 7.4 vLLM 추론

속도가 더 필요하면 `infer_backend: vllm`을 사용한다.

```yaml
infer_backend: vllm
```

배치 추론 예시는 다음과 같다.

```bash
python scripts/vllm_infer.py \
    --model_name_or_path path_to_merged_model \
    --dataset alpaca_en_demo
```

---

## 8. 평가

### 8.1 일반 능력 평가

```bash
llamafactory-cli eval examples/train_lora/llama3_lora_eval.yaml
```

평가 설정 예시의 핵심 항목은 다음과 같다.

- `model_name_or_path`
- `adapter_name_or_path`(선택)
- `finetuning_type`
- `task: mmlu_test | ceval_validation | cmmlu_test`
- `template: fewshot`
- `lang: en | zh`
- `n_shot`
- `save_dir`

### 8.2 NLG 평가

생성 품질을 보려면 예측 모드 설정을 사용한다.

```bash
llamafactory-cli train examples/extras/nlg_eval/llama3_lora_predict.yaml
```

필요하면 병합 모델에 대해 `vllm_infer.py`를 사용해 빠르게 비교할 수도 있다.

---

## 9. LoRA 병합 및 내보내기

학습이 끝난 뒤 베이스 모델과 어댑터를 하나의 완성 모델로 합치려면 `export`를 사용한다.

`configs/merge_my_sft.yaml`

```yaml
### model
model_name_or_path: your_model_name_or_path
adapter_name_or_path: saves/my_model/lora/sft
template: your_template
finetuning_type: lora

### export
export_dir: models/my_model_sft
export_size: 2
export_device: cpu
export_legacy_format: false
```

실행:

```bash
llamafactory-cli export configs/merge_my_sft.yaml
```

### 9.1 병합 시 주의사항

- `model_name_or_path`는 실제로 존재해야 한다.
- `template`는 학습 때 사용한 값과 일치해야 한다.
- `adapter_name_or_path`는 학습 때의 `output_dir`와 일치해야 한다.
- **양자화된 모델로 LoRA를 병합하지 않는다.** 병합 단계에서는 비양자화 베이스 모델을 사용한다.

병합이 끝나면 `export_dir` 아래의 모델을 일반 Hugging Face 형식처럼 배포하거나 추론에 사용할 수 있다.

---

## 10. WebUI

코드 대신 GUI로 진행하려면 WebUI를 실행한다.

```bash
llamafactory-cli webui
```

Training 화면에서 먼저 지정할 항목은 다음과 같다.

1. 모델 이름 및 경로
2. 학습 단계
3. 미세조정 방식
4. 학습 데이터셋
5. 학습률, epoch 등 학습 파라미터
6. 미세조정 파라미터 및 기타 옵션
7. 출력 디렉터리와 설정 파일 경로

커스텀 데이터셋을 쓰는 경우에도, 먼저 `data/dataset_info.json`에 등록되어 있어야 한다.

체크포인트를 이어서 학습할 때는 `output_dir`에 저장된 어댑터 체크포인트를 로드해 재개한다.

---

## 11. 문제 해결

### 11.1 `Cannot open data/dataset_info.json`

다음 항목을 확인한다.

- `data/dataset_info.json` 파일이 실제로 존재하는지
- JSON 문법 오류가 없는지
- 작업 디렉터리가 LLaMA-Factory 루트인지

### 11.2 `Cannot find valid samples`

대부분 데이터 형식 문제다.

- 데이터 파일의 키 이름과 `dataset_info.json`의 `columns`가 일치하는지
- Alpaca 형식인데 `instruction`, `output`이 빠지지 않았는지
- ShareGPT 형식인데 `formatting: sharegpt`가 선언되어 있는지
- 문자열이어야 할 필드에 배열이나 `null`이 들어가지 않았는지

### 11.3 `KeyError: 'instruction'`

보통 다음 중 하나다.

- 실제 데이터는 ShareGPT/OpenAI 형식인데 Alpaca처럼 등록함
- `formatting: sharegpt`가 빠짐
- `columns` 매핑이 잘못됨

### 11.4 모델이 엉뚱하게 답하거나 응답이 이상함

가장 먼저 `template`를 확인한다.

- 모델과 템플릿이 맞는지
- 학습과 추론에서 템플릿이 같은지
- Qwen3처럼 `qwen3`와 `qwen3_nothink`를 구분해야 하는 모델은 아닌지

### 11.5 학습이 너무 무겁거나 메모리가 부족함

다음 순서로 줄인다.

1. `per_device_train_batch_size`를 낮춘다.
2. `gradient_accumulation_steps`로 보정한다.
3. `cutoff_len`을 줄인다.
4. `max_samples`로 소규모 검증부터 한다.
5. 필요하면 LoRA 대신 QLoRA를 고려한다.

### 11.6 원하는 GPU만 쓰고 싶음

기본적으로 LLaMA-Factory는 보이는 모든 연산 장치를 사용한다. 특정 장치만 쓰려면 환경 변수를 지정한다.

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/train_my_sft.yaml
```

Ascend NPU 환경에서는 `ASCEND_RT_VISIBLE_DEVICES`를 사용한다.

### 11.7 병합이 실패함

다음 항목을 확인한다.

- 병합 시 양자화 모델을 쓰지 않았는지
- `adapter_name_or_path`가 학습 결과 디렉터리와 같은지
- `template`가 학습 때와 같은지

---

## 12. 다음 단계

첫 번째 SFT가 끝나면 보통 아래 순서로 확장한다.

1. **QLoRA**: VRAM이 부족할 때
2. **Freeze / Full**: 더 큰 수정이 필요할 때
3. **DPO / ORPO / KTO**: 선호 학습과 정렬이 필요할 때
4. **vLLM API**: 서비스 속도가 중요할 때
5. **DeepSpeed / FSDP / DDP**: 단일 GPU를 넘어갈 때

LoRA와 QLoRA만으로도 대부분의 초기 실험은 충분히 진행할 수 있다. 처음부터 full fine-tuning과 분산 학습까지 함께 잡기보다, 단일 GPU에서 SFT 한 번을 안정적으로 끝내는 편이 전체 시행착오를 크게 줄여 준다.

---

## 13. 참고 문서

- 공식 문서 홈: <https://llamafactory.readthedocs.io/en/latest/>
- Installation: <https://llamafactory.readthedocs.io/en/latest/getting_started/installation.html>
- Data Preparation: <https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html>
- WebUI: <https://llamafactory.readthedocs.io/en/latest/getting_started/webui.html>
- Supervised Fine-tuning: <https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html>
- Inference: <https://llamafactory.readthedocs.io/en/latest/getting_started/inference.html>
- Eval: <https://llamafactory.readthedocs.io/en/latest/getting_started/eval.html>
- LoRA Merge: <https://llamafactory.readthedocs.io/en/latest/getting_started/merge_lora.html>
- Tuning Algorithms: <https://llamafactory.readthedocs.io/en/latest/advanced/tuning_algorithms.html>
- Distributed Training: <https://llamafactory.readthedocs.io/en/latest/advanced/distributed.html>
- GitHub README: <https://github.com/hiyouga/LLaMA-Factory>
- examples/README.md: <https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/README.md>
- data/README.md: <https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md>
