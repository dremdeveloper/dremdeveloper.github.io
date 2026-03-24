# Nemotron-Cascade-RL-Instruction-Following 데이터셋
## 1. 개요
Nemotron-Cascade-RL-Instruction-Following은 NVIDIA가 공개한 **Instruction-Following Reinforcement Learning(IF-RL)** 용 데이터셋이다.  
모델이 사용자의 다양한 제약 조건을 **정확히 지키는지**를 학습·평가하기 쉽도록, 일반적인 `질문-정답` 쌍보다 **프롬프트 + 규칙 검증용 메타데이터(rule-verifier metadata)** 중심으로 설계되어 있다.

핵심은 **정답 문장 자체를 외워 맞히는 데이터셋**이 아니라, **생성한 응답이 규칙을 만족하는지 자동으로 판정할 수 있는 데이터셋**이라는 점이다.  
예를 들어 제목 포함, bullet 수, 단어 수 제한, 종료 문구, 소문자 강제, 키워드 포함 여부 같은 제약을 verifier로 검사하는 구조에 적합하다.

---

## 2. 크기

| 구분 | 내용 |
|---|---|
| 전체 샘플 수 | 108,938 |
| 공개 Split | train |
| Hugging Face 크기 구간 | 100K–1M |
| 전체 디스크 크기 | 약 26 MB |
| 자동 변환 Parquet 다운로드 크기 | 24.1 MB |
| 언어 | English |
| 공개 시점 | 2025-12-15 |

---

## 3. 형식

| 항목 | 내용 |
|---|---|
| 데이터 모달리티 | Text |
| 저장 포맷 | Parquet |
| 데이터 구조 | Text + Metadata |
| 주요 컬럼 | `prompt`, `instruction_id_list`, `kwargs`, `index` |
| `prompt` 형식 | Chat format |
| `instruction_id_list` 의미 | rule verifier가 적용할 instruction-following 규칙 ID 목록 |
| `kwargs` 의미 | 각 규칙 검증에 필요한 파라미터 메타데이터 |
| `index` 의미 | 샘플 식별자 |
| 데이터 수집 방식 | Hybrid: Human, Synthetic, Automated |
| 라벨링 방식 | Hybrid: Human, Synthetic, Automated |
| 라이선스 | ODC-BY-1.0 |
| 상업적 사용 | 가능(출처 표기 조건) |

---

## 4. Raw Data 1개

### 미리보기 발췌
```json
{
  "prompt": [
    {
      "content": "Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>. Answer with less than 100 words.\n",
      "role": "user"
    }
  ],
  "instruction_id_list": [
    "detectable_format:title",
    "length_constraints:number_words"
  ],
  "index": 0
}
```

> 참고: 실제 원본 row에는 `kwargs`도 함께 존재한다. Hugging Face viewer에서는 `kwargs` 객체가 길어서 축약되어 보이는 경우가 많다.

### Raw Data 설명

| 항목 | 설명 |
|---|---|
| `prompt` | 모델 입력용 사용자 지시문이다. chat list 구조라서 chat template 기반 학습/추론 파이프라인에 바로 연결하기 쉽다. |
| `instruction_id_list` | 어떤 규칙으로 응답을 검사할지 정의하는 verifier ID 목록이다. 예: 제목 포함, bullet 개수, 키워드 포함, 종료 문구, 소문자 강제 등 |
| `kwargs` | 각 규칙 검사에 필요한 파라미터 묶음이다. viewer 미리보기 기준으로 `end_phrase`, `keyword`, `keywords`, `forbidden_words`, `letter`, `nth_paragraph`, `num_*` 계열 값들이 들어가는 형태다. |
| `index` | 샘플 식별자이다. 디버깅, 실패 케이스 추적, 재현 실험에 사용하기 좋다. |

### Raw Data 해석 방법

| 해석 포인트 | 의미 |
|---|---|
| 이 row가 요구하는 것 | 제목을 `<< >>` 형태로 포함하고, 전체 응답을 단어 수 제한 안에 유지해야 한다. |
| 평가 방식 | 모델이 응답을 생성한 뒤, 각 verifier가 규칙별로 통과/실패를 판정한다. |
| 보상 설계 연결 | 규칙별 결과를 개별 점수로 만들고, 이를 합산해 RL 보상으로 사용할 수 있다. |
| 실무 해석 | 이 데이터는 answer-supervision보다 **constraint-supervision** 성격이 강하다. |

### Raw Data의 구조적 의미
- 한 row는 **프롬프트 1개 + 복수의 출력 제약 조건**으로 이해하면 된다.
- `instruction_id_list`와 `kwargs`는 보통 같은 위치의 항목끼리 대응해 사용한다.
- 따라서 학습이나 평가에서는 **응답 생성 → 규칙별 검증 → 개별 점수/총점 계산** 흐름으로 연결된다.

---

## 5. 특징

| 구분 | 설명 |
|---|---|
| 목적 특화 | 일반 대화용 데이터가 아니라 **strict instruction following** 강화를 위한 RL 데이터셋이다. |
| 검증 가능성 중심 설계 | 응답 길이, 형식, 키워드, 종료 구문, 대소문자 등 **프로그램으로 검증 가능한 제약** 위주로 구성된다. |
| 자동 채점 친화성 | `instruction_id_list`와 `kwargs`가 함께 제공되어 rule-based verifier와 바로 연결하기 좋다. |
| 복합 제약 대응 | 한 프롬프트에 여러 제약을 동시에 걸 수 있어, 단일 규칙뿐 아니라 **조합 제약 준수 능력**도 다룰 수 있다. |
| 데이터 출처 확장성 | 필터링된 **Llama-Nemotron-Post-Training-Dataset**과 **LMSYS-Chat-1M** 프롬프트 증강을 활용해 다양성을 확보했다. |
| 학습·평가 겸용성 | 학습용 RL prompt로도 쓰기 좋고, verifier 기반의 내부 평가 세트로도 활용하기 좋다. |
| 활용 범위 | instruction adherence, structured output, constraint following, verifier-based RL, formatting control 실험에 적합하다. |
| 공개 활용성 | 공개 데이터셋이며 ODC-BY-1.0 조건 아래 재학습·평가에 활용 가능하다. |

---

## 6. 해당 데이터셋으로 평가했을 때 확인할 수 있는 것

| 평가 관점 | 확인 가능한 내용 | 해석 |
|---|---|---|
| Prompt-level strict pass | 한 응답이 **해당 프롬프트의 모든 제약을 동시에 만족했는지** 확인 가능 | 실제 사용자 지시를 끝까지 지키는지 보는 가장 엄격한 기준 |
| Instruction-level pass | 복합 지시문 안에서 **어떤 규칙을 지켰고 어떤 규칙을 어겼는지** 분해 가능 | 형식은 맞지만 종료 문구는 틀리는 식의 실패 유형 분리 가능 |
| 형식 제어 능력 | 제목, bullet, 강조, paragraph 구분, placeholder 등 **출력 외형 제어 능력** 확인 가능 | structured output 안정성 점검에 유용 |
| 길이 제어 능력 | 단어 수, 문장 수, 문단 수, section 수 같은 **길이/분량 제어 능력** 확인 가능 | 장황함, 과도한 설명, 짧은 응답 문제 분석 가능 |
| 내용 제약 준수 능력 | 특정 keyword 포함, forbidden word 배제, 특정 표현 유지 여부 확인 가능 | 내용 조건을 얼마나 정밀하게 따르는지 파악 가능 |
| 표현 통제 능력 | lowercase, 특정 스타일, 특정 마감 문구 등 **표현 방식 제어 능력** 확인 가능 | tone/style보다 더 강한 rule-following 통제력 확인 가능 |
| 복합 규칙 일반화 | 단일 규칙은 잘 따르지만 **여러 규칙을 동시에 걸면 무너지는지** 확인 가능 | compositional instruction following 강약 파악 가능 |
| 실패 패턴 분포 | 모델이 주로 길이, 형식, 키워드, 종료 패턴 중 어디에서 자주 실패하는지 확인 가능 | reward shaping, curriculum, data augmentation 방향 설정 가능 |

### 실무 해석 기준

| 기준 | 의미 |
|---|---|
| Prompt strict | 프롬프트 안의 제약을 **전부 만족해야 통과**로 보는 기준 |
| Instruction strict | 각 규칙을 개별 단위로 나눠서 통과율을 계산하는 기준 |
| In-domain adherence | 같은 taxonomy, 같은 verifier 체계 안에서 얼마나 잘 따르는지 보는 기준 |
| Failure mode analysis | 어떤 verifier 유형에서 자주 실패하는지 분해해 보는 기준 |

### 평가 시 주의할 점
이 데이터셋 자체로 평가하면 주로 **in-domain precise instruction following** 능력을 확인할 수 있다.  
반면 **보지 못한 새로운 constraint taxonomy에 대한 일반화**까지 확인하려면 별도의 held-out benchmark가 필요하다. 이런 목적의 대표 예가 **IFBench**이며, IFBench는 IFEval에 없는 **58개의 새로운 verifiable constraint**로 일반화 성능을 평가하도록 설계되어 있다.

---

## 7. veRL에서 수행했을 때 `cumstore_score` 반환값 해석

### 7-1. 먼저 확인할 점
`cumstore_score`라는 이름은 **veRL 공식 문서의 표준 반환 필드명으로 직접 문서화되어 있지 않다.**  
공개 veRL mainline에서 custom reward의 기본 진입점은 보통 `compute_score`이며, reward manager는 이 값을 받아 `score`와 추가 정보들을 기록하는 방식으로 동작한다.

즉, `cumstore_score`는 대개 **프로젝트 내부에서 `compute_score` 결과를 누적 저장할 때 붙인 변수명**으로 해석하는 것이 맞다.

### 7-2. 공개 veRL / GDPO 예시 기준 의미
NVIDIA의 공개 **GDPO veRL 기반 예시**에서는 reward function이 아래 4개 값을 반환한다.

```text
(score, format_score, correctness_score, length_score)
```

### 7-3. 각 값의 의미

| 값 | 의미 | 실무 해석 |
|---|---|---|
| `score` | 최종 총합 점수 | 일반적으로 `format_score + correctness_score + length_score`의 합 |
| `format_score` | 형식 준수 점수 | 출력 구조, 태그, 제목, bullet, section, 종료 형식 등 **응답 외형 규칙**의 준수 여부 |
| `correctness_score` | 정합성/정확성 점수 | 요구된 핵심 내용이 맞는지에 대한 점수. 구현에 따라 정답 일치, tool call 정합성, parameter match 등이 포함됨 |
| `length_score` | 길이 제어 점수 | 응답 길이 또는 reasoning 길이에 대한 보상. 설정에 따라 0으로 비활성화될 수 있음 |

### 7-4. 공개 구현에서 실제로 어떻게 쓰이는가

| 구현 위치 | 동작 |
|---|---|
| GDPO `rlla.py` | `compute_score`가 `(score, fomrat_score, correctness_score, length_score)`를 반환 |
| GDPO `main_ppo.py` | 반환된 4개 값을 `reward_tensor`, `format_tensor`, `correctness_tensor`, `length_tensor`에 각각 저장 |
| GDPO `ray_trainer.py` | `token_level_scores`, `token_level_scores_format`, `token_level_scores_correctness`, `token_level_scores_length`로 분리 기록 |
| veRL mainline validation | `acc`가 없으면 기본 핵심 검증 지표는 `reward` 기준으로 집계 |

### 7-5. 실무적으로 읽는 방법
프로젝트 내부에서 `cumstore_score`가 위 4개 값을 누적 저장하는 컨테이너라면, 일반적으로 아래 순서로 해석하면 된다.

```text
[총합 점수, 형식 점수, 정합성/정확성 점수, 길이 점수]
```

### 7-6. 해석 시 주의점
- `cumstore_score` 자체는 veRL mainline의 공식 표준 변수명이 아니다.
- **정확한 필드명과 순서**는 사용 중인 reward function 구현에 따라 달라질 수 있다.
- 가장 정확한 확인 방법은 실제 코드에서 **`compute_score` 반환부**와 **trainer logging 키**를 함께 확인하는 것이다.

---

## 8. 한줄 정리
Nemotron-Cascade-RL-Instruction-Following은 **모델이 요구된 방식대로 답했는지**를 길이·형식·키워드·표현 제약 수준에서 자동 검증할 수 있도록 설계된, **NVIDIA의 verifier-friendly IF-RL 데이터셋**이다.

---

## 9. 출처
- NVIDIA Hugging Face Dataset Card: `nvidia/Nemotron-Cascade-RL-Instruction-Following`
- NVIDIA Technical Report: *Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation*
- Technical Report: *Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models*
- Google / Yale: *Instruction-Following Evaluation for Large Language Models (IFEval)*
- AllenAI / UW: *Generalizing Verifiable Instruction Following (IFBench)*
- veRL Documentation: *Implement Reward Function for Dataset*, *Reward Loop*
- veRL Source: `verl/workers/reward_manager/naive.py`, `verl/trainer/ppo/ray_trainer.py`
- NVIDIA GDPO Source: `verl-GDPO/verl/trainer/main_ppo.py`, `verl-GDPO/verl/utils/reward_score/rlla.py`, `verl-GDPO/verl/trainer/ppo/ray_trainer.py`
