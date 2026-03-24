# GSM8K 데이터셋

## 1. 개요
GSM8K(Grade School Math 8K)는 초등~중등 수준의 수학 서술형 문제를 모아 만든 대표적인 수학 추론 벤치마크 데이터셋이다. 기본 산술 연산과 다단계 추론 능력을 평가하는 데 주로 사용된다.

## 2. 크기

| 구성 | Train | Test | 합계 |
|---|---:|---:|---:|
| `main` | 7,473 | 1,319 | 8,792 |
| `socratic` | 7,473 | 1,319 | 8,792 |

> 일반적으로 벤치마크와 학습/평가 예시에서는 `main` 구성을 가장 많이 사용한다.

## 3. 형식

| 항목 | 내용 |
|---|---|
| 데이터 단위 | 1문항당 1개 레코드 |
| 원본 저장 형식 | JSONL (`train.jsonl`, `test.jsonl`) |
| 배포 형식 | Parquet(Hugging Face 배포본 기준) |
| 주요 필드 | `question`, `answer` |
| `question` | 수학 서술형 문제 텍스트 |
| `answer` | 단계별 풀이 + 계산 annotation + 최종 정답 |
| 정답 표기 규칙 | 마지막 줄에 `#### 숫자` 형식으로 최종 답을 기록 |
| 계산 annotation 예시 | `<<48/2=24>>` |
| 언어 | English |
| 구성 | `main`, `socratic` |
| `main` 특징 | 일반 단계별 풀이가 포함된 기본 구성 |
| `socratic` 특징 | 풀이 단계 사이에 소크라테스식 하위 질문이 삽입된 구성 |

## 4. Raw Data 예시

```json
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
}
```

## 5. Raw Data 설명

- `question`은 모델에 입력되는 문제 문장이다.
- `answer`는 단순 정답만 들어 있는 필드가 아니라, 자연어 풀이 과정과 계산 annotation, 최종 정답을 함께 담는 필드다.
- `<<48/2=24>>` 같은 표기는 계산 annotation이다. 식과 계산 결과를 함께 남겨 계산기 호출 또는 후처리 규칙과 연결하기 쉽다.
- 마지막 줄의 `#### 72`는 최종 정답 추출용 앵커 역할을 한다. 평가 시에는 이 마지막 숫자를 기준으로 정답을 비교하는 경우가 많다.
- 즉, GSM8K의 raw data는 **문제(`question`) + 풀이와 정답이 함께 포함된 정답 문자열(`answer`)** 구조이며, 별도의 독립 숫자 라벨 필드가 따로 분리되어 있지 않다.

## 6. 특징

- 초등~중등 수준의 수학 문장제를 중심으로 구성되어 있다.
- 한 문제를 풀기 위해 보통 2~8단계의 추론이 필요하다.
- 덧셈, 뺄셈, 곱셈, 나눗셈 같은 기본 산술 연산이 중심이다.
- 정답만 있는 형식이 아니라, 자연어 기반의 풀이 과정까지 함께 제공한다.
- 계산 과정을 `<<식=값>>` 형태로 표시해 계산기 연동이나 정답 추출에 활용하기 쉽다.
- 마지막 줄의 `#### 정답` 패턴 덕분에 자동 평가가 비교적 단순하다.
- LLM의 수학 추론, final answer extraction, chain-of-thought 스타일 응답 구조를 함께 다루기 좋은 데이터셋이다.
- 영어 단일 언어 데이터셋이므로 다국어 수학 추론 성능을 직접 대표하지는 않는다.

## 7. 이 데이터셋으로 평가했을 때 확인할 수 있는 것

- **최종 숫자 정답 정확도**: 모델이 문장제의 최종 답을 정확히 산출하는지 확인할 수 있다.
- **다단계 산술 추론 능력**: 2~8단계 정도의 계산과 추론이 필요한 문제를 얼마나 안정적으로 푸는지 확인할 수 있다.
- **문장제 해석 능력**: 단순 계산뿐 아니라 수량 관계, 조건, 순서를 올바르게 읽고 식으로 바꾸는 능력을 확인할 수 있다.
- **출력 형식 준수 여부**: `#### 정답` 형식을 요구하는 평가/보상 함수에서는 모델이 지정된 답안 형식을 지키는지도 함께 확인할 수 있다.
- **다중 샘플 평가 시 응답 안정성**: 한 문제에 대해 여러 개의 응답을 생성하면 평균 성능(`mean@N`), 최고 응답 잠재력(`best@N`), 최악 응답 수준(`worst@N`), 다수결 기반 일관성(`maj@N`)까지 볼 수 있다.
- **한계**: 기본 GSM8K는 주로 최종 정답 중심으로 평가되므로, 중간 풀이의 논리적 타당성이나 설명 품질 자체를 정밀하게 채점하는 벤치마크는 아니다.

## 8. veRL 기준 `compute_score` 반환값 의미

veRL 공식 문서 기준 사용자 정의 보상 함수의 기본 이름은 `compute_score`다. 학습 시 이 함수는 보통 `data_source`, `solution_str`, `ground_truth`, `extra_info`를 입력으로 받아 보상 점수를 계산한다.

### 8.1 사용자 정의 `compute_score`가 직접 반환하는 값

| 반환 형태 | 의미 | 실무 해석 |
|---|---|---|
| `float` 또는 `int` | 최종 scalar reward | 가장 단순한 형태. 예: 정답이면 `1.0`, 오답이면 `0.0` |
| `{"score": x}` | 최종 scalar reward를 명시적으로 반환 | dict를 쓸 경우 `score`가 핵심 보상값 |
| `{"score": x, "acc": y}` | 보상과 정확도 지표를 분리 | `x`는 학습 보상, `y`는 검증용 accuracy 성격 지표로 활용 가능 |
| `{"score": x, "pred": z}` | 보상과 모델의 예측 답안을 함께 반환 | `pred`가 있으면 validation에서 다수결 기반 `maj@N` 지표 계산에 활용 가능 |
| `{"score": x, "기타키": ...}` | 추가 로깅 지표 포함 | 예: `format_reward`, `tool_reward`, `reasoning_score` 같은 사용자 정의 메타릭 |

### 8.2 veRL 내부에서 바뀌는 값

| 내부 키 | 의미 |
|---|---|
| `reward_tensor` | 토큰 단위 보상 텐서. 실제 reward는 보통 **응답의 마지막 유효 토큰 위치**에 기록된다. |
| `reward_extra_info` | `score` 외의 부가 지표들을 샘플별로 모아둔 dict |
| `reward_score` | agent loop / reward loop 비동기 경로에서 쓰는 **단일 샘플 scalar reward** |
| `rm_scores` | `reward_score`가 배치 형태 텐서로 정리된 값 |

### 8.3 validation에서 자주 보게 되는 지표 이름

| 지표 | 의미 |
|---|---|
| `mean@N` | 같은 문제에 대해 N개 응답을 생성했을 때 평균 점수 |
| `std@N` | N개 응답 점수의 분산 정도 |
| `best@N/mean` | N개 중 가장 좋은 응답을 고를 수 있다고 가정했을 때의 기대 점수 |
| `worst@N/mean` | N개 중 가장 나쁜 응답 기준 기대 점수 |
| `maj@N/mean` | `pred`를 기준으로 다수결을 했을 때의 기대 점수 |

### 8.4 해석 시 주의할 점

- validation 결과에 `acc`가 있으면 veRL은 `acc`를 core metric으로 우선 표시하고, `acc`가 없으면 `reward`를 core metric으로 사용한다.
- 최신 `naive` reward manager는 scalar만 반환하면 그 값을 reward로만 사용한다.
- 최신 `dapo` reward manager는 scalar만 반환한 경우 그 값을 `acc`에도 함께 기록할 수 있어 검증 로그에서 정확도처럼 보이게 만들 수 있다.
- `pred`를 반환하지 않으면 `maj@N` 계열 지표는 계산되지 않는다.
- agentic/multi-turn 설정에서는 `__num_turns__`가 함께 수집되어 문제당 몇 번의 상호작용이 있었는지도 별도 메트릭으로 볼 수 있다.

## 9. 참고 자료

- [Hugging Face - openai/gsm8k Dataset Card](https://huggingface.co/datasets/openai/gsm8k)
- [OpenAI grade-school-math Repository](https://github.com/openai/grade-school-math)
- [veRL Documentation - Implement Reward Function for Dataset](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
- [veRL Documentation - Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)
- [veRL Source - `verl/workers/reward_manager/naive.py`](https://verl.readthedocs.io/en/latest/_modules/verl/workers/reward_manager/naive.html)
- [veRL Source - `verl/workers/reward_manager/dapo.py`](https://verl.readthedocs.io/en/latest/_modules/verl/workers/reward_manager/dapo.html)
- [veRL Source - `verl/trainer/ppo/metric_utils.py`](https://raw.githubusercontent.com/verl-project/verl/main/verl/trainer/ppo/metric_utils.py)
- [veRL Source - `verl/experimental/agent_loop/agent_loop.py`](https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/agent_loop.py)
