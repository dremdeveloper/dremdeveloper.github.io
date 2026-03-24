# IF-Eval-VERL 데이터셋

대상 데이터셋: `sungyub/ifeval-rlvr-verl`  
정식 명칭: `RLVR-IFeval-VERL`  
원본 데이터셋: `allenai/RLVR-IFeval`

`IF-Eval-VERL`은 원본 RLVR-IFeval 데이터를 VERL 학습 및 평가 파이프라인에서 바로 사용할 수 있도록 정리한 instruction-following 데이터셋이다.

## 1. 크기

| 항목 | 내용 |
|---|---|
| 데이터셋 ID | `sungyub/ifeval-rlvr-verl` |
| 원본 데이터셋 | `allenai/RLVR-IFeval` |
| 총 샘플 수 | 14,973 |
| Split | `train` 1개 |
| 파일 형식 | `Parquet` |
| 데이터 크기 | 11,309,939 bytes (약 11.3 MB) |
| 언어 | English |
| 제약 유형 수 | 25개 |
| 제약 카테고리 수 | 9개 |

## 2. 형식

### 2-1. 레코드 스키마

| 필드 | 타입 | 설명 |
|---|---|---|
| `data_source` | `string` | 원본 소스 식별자. 값은 `allenai/IF_multi_constraints_upto5`로 고정 |
| `prompt` | `list` | 대화형 입력 메시지 목록 |
| `prompt[].role` | `string` | 메시지 역할. 일반적으로 `user` |
| `prompt[].content` | `string` | 실제 지시문과 제약 조건이 들어 있는 본문 |
| `ability` | `string` | 태스크 범주. 값은 `instruction_following`으로 고정 |
| `reward_model` | `dict` | 자동 평가용 보상 모델 설정 |
| `reward_model.style` | `string` | 평가 스타일. 값은 `ifeval`로 고정 |
| `reward_model.ground_truth` | `string` | 제약 조건 정답이 들어 있는 Python literal string |
| `extra_info` | `dict` | 부가 메타데이터 |
| `extra_info.index` | `int64` | 원본 데이터 인덱스 |
| `dataset` | `string` | 데이터셋 식별자. 값은 `ifeval`로 고정 |

### 2-2. 제약 카테고리 구성

| 카테고리 | 유형 수 | 대표 제약 |
|---|---:|---|
| Keywords | 4 | 특정 키워드 포함, 키워드 빈도, 금지어, 특정 문자 빈도 |
| Language | 1 | 응답 언어 지정 |
| Length Constraints | 4 | 문단 수, 단어 수, 문장 수, n번째 문단 시작 단어 |
| Detectable Content | 2 | postscript 포함, placeholder 개수 |
| Detectable Format | 6 | bullet list 수, title, constrained response, highlighted section 수, multiple sections, JSON 형식 |
| Combination | 2 | prompt 반복, 두 개의 응답 생성 |
| Case Changes | 3 | 전체 대문자, 전체 소문자, 대문자 단어 빈도 |
| Start/End | 2 | 특정 문구로 끝내기, 따옴표로 감싸기 |
| Punctuation | 1 | 쉼표 사용 금지 |

## 3. rawdata 1개

```json
{
  "data_source": "allenai/IF_multi_constraints_upto5",
  "prompt": [
    {
      "role": "user",
      "content": "Describe the purpose of internet protocol version 6 (IPv6).\n\n Your entire response should be in English, and in all lowercase letters. No capital letters are allowed."
    }
  ],
  "ability": "instruction_following",
  "reward_model": {
    "style": "ifeval",
    "ground_truth": "[{'instruction_id': ['change_case:english_lowercase'], 'kwargs': [None]}]"
  },
  "extra_info": {
    "index": 0
  },
  "dataset": "ifeval"
}
```

### 3-1. rawdata 설명

| 항목 | 설명 | 이 예시에서 읽을 수 있는 의미 |
|---|---|---|
| `data_source` | scorer 라우팅과 소스 식별에 쓰이는 값 | 어떤 보상 함수와 데이터 계열인지 식별하는 키로 사용 |
| `prompt` | 모델에 입력되는 실제 대화형 프롬프트 | IPv6 설명이라는 본작업 + 소문자/영문 응답이라는 제약이 함께 들어 있다. |
| `ability` | 태스크 성격을 나타내는 고정 필드 | 이 샘플은 일반 QA가 아니라 `instruction_following` 평가용 샘플이다. |
| `reward_model.style` | 어떤 평가기를 쓸지 지정하는 값 | `ifeval` scorer로 채점해야 함을 뜻한다. |
| `reward_model.ground_truth` | 정답 문장 자체가 아니라, 검증해야 할 제약 규칙을 저장한 문자열 | 이 예시는 `change_case:english_lowercase` 제약 1개를 만족해야 한다는 뜻이다. |
| `extra_info.index` | 원본 샘플 인덱스 | 원본 데이터와 역추적하거나 디버깅할 때 기준점으로 쓸 수 있다. |
| `dataset` | 데이터셋 이름 태그 | scorer와 분석 코드에서 데이터셋 그룹을 구분할 때 사용한다. |

### 3-2. ground_truth 해석

| 구성 요소 | 값 | 의미 |
|---|---|---|
| `instruction_id` | `change_case:english_lowercase` | 응답 전체를 영문 소문자로 작성해야 하는 제약 |
| `kwargs` | `None` | 추가 숫자 파라미터나 옵션이 없는 단순 제약 |
| 저장 방식 | Python literal string | 파싱 후 제약 검사기 입력으로 넘기기 쉬운 형태 |

### 3-3. rawdata를 볼 때 핵심 포인트

| 포인트 | 설명 |
|---|---|
| gold answer 텍스트가 직접 들어 있지 않음 | 이 데이터셋은 정답 문장을 저장하기보다, 응답이 따라야 할 규칙을 저장한다. |
| 프롬프트 안에 본작업과 제약이 함께 존재 | 모델은 내용 생성과 형식 준수를 동시에 만족해야 한다. |
| 자동 채점 친화적 구조 | `prompt` + `reward_model.style` + `ground_truth`만으로 scorer 호출과 검증 규칙 적용이 가능하다. |

## 4. 특징

| 특징 | 설명 |
|---|---|
| VERL 바로 연동 가능 | `prompt`, `ability`, `reward_model`, `extra_info` 구조를 그대로 VERL 파이프라인에 연결하기 쉽다. |
| 검증 가능한 instruction-following 평가 | 응답 제약이 자연어 지시문과 `ground_truth`에 함께 표현되어 자동 채점이 가능하다. |
| 제약 기반 성능 분석에 적합 | 25개 제약 유형을 9개 카테고리로 나눠 모델의 약점을 세부적으로 분석할 수 있다. |
| RLVR용 변환 데이터 | 원본 `allenai/RLVR-IFeval`을 IFBench-VERL 스키마에 맞게 변환한 버전이다. |
| 스키마가 단순하고 일관적임 | 핵심 필드 수가 적고, 상수 값 필드가 많아 전처리와 샘플링이 쉽다. |
| 평가 결과 해석이 명확함 | scorer 결과와 veRL 로그를 분리해서 보면 실제 보상과 보조 형식 지표를 함께 확인할 수 있다. |
| 응답 포맷 호환성 보유 | XML 형식과 GPT-OSS 형식을 모두 지원하며 자동 감지가 가능하다. |
| 텍스트 기반 IF 테스트셋 | 영어 텍스트 중심의 instruction-following 능력 측정에 집중된 구성이다. |

## 5. 해당 데이터셋으로 평가했을 때 확인할 수 있는 것

| 확인 항목 | 설명 |
|---|---|
| 전체 instruction-following 준수 정도 | `score`를 통해 각 응답이 지시문과 제약을 얼마나 충족했는지 확인할 수 있다. |
| 응답 포맷 적합성 | `reward_fmt`를 통해 scorer가 기대하는 응답 구조가 맞는지 별도로 볼 수 있다. |
| reasoning 구간 적합성 | `reward_think`를 통해 reasoning 또는 think 구간이 기대 형식에 맞는지 확인할 수 있다. |
| 제약 유형별 강점/약점 | 25개 constraint type, 9개 category 단위로 어느 유형에서 잘 따르고 어느 유형에서 자주 실패하는지 분석할 수 있다. |
| 동일 프롬프트에 대한 샘플링 안정성 | veRL validation에서 여러 응답을 생성하면 `mean@N`, `std@N`, `best@N/mean`, `worst@N/mean`으로 평균 성능과 변동성을 확인할 수 있다. |
| best-of-N 잠재력 | 같은 프롬프트에 여러 응답을 샘플링했을 때 최고 성능이 얼마나 올라가는지 확인할 수 있다. |
| 실제 학습용 핵심 지표와 보조 지표 구분 | veRL에서는 실제 보상으로 쓰이는 핵심 값과 포맷/think 같은 보조 값을 분리해서 해석할 수 있다. |
| 평가 범위의 성격 | 이 데이터셋은 주로 명시적 제약 준수 여부를 보는 데 강하고, 사실 정확성이나 도메인 지식의 깊이 자체는 별도 벤치마크로 함께 보는 것이 적합하다. |

## 6. veRL에서 확인하는 scorer 반환값 의미

| 항목 | 의미 | veRL 내부에서의 해석 |
|---|---|---|
| `score` | instruction-following 원점수 | RewardManager가 이 값을 꺼내 실제 `reward_score`로 사용한다. |
| `reward_fmt` | 응답 구조/형식 적합 여부를 보여주는 보조 점수 | `reward_extra_info`로 보존되고, validation에서는 보조 지표로 확인한다. |
| `reward_think` | reasoning 또는 `<think>` 구간의 구조 적합 여부를 보여주는 보조 점수 | `reward_extra_info`로 보존되고, validation에서는 보조 지표로 확인한다. |
| `reward` | veRL validation/logging에서 집계되는 핵심 보상 값 | 위 `score`가 실제 보상으로 반영된 뒤 `val-core` 계열에서 주로 보이는 값이다. |

### 6-1. 실무 해석 포인트

| 포인트 | 설명 |
|---|---|
| scorer의 반환 딕셔너리와 veRL 로그 이름은 1:1로 같지 않을 수 있음 | scorer는 `score`, `reward_fmt`, `reward_think`를 반환하더라도, veRL 핵심 로그에서는 보통 `reward`가 중심 지표가 된다. |
| `score`는 실제 최적화에 들어가는 값 | dict를 반환하는 scorer를 사용할 때 veRL은 `result['score']`를 실제 보상으로 사용한다. |
| `reward_fmt`, `reward_think`는 디버깅용 가치가 큼 | 학습이 안 되는 원인이 내용 문제인지, 응답 구조 문제인지 빠르게 분리해서 볼 수 있다. |
| validation에서는 `val-core`와 `val-aux`를 같이 봐야 함 | `val-core`는 실제 성능 판단용, `val-aux`는 형식 실패나 think 블록 실패 원인 진단용으로 해석하면 된다. |

### 6-2. veRL 로그 이름 예시 해석

| 예시 로그 이름 | 해석 |
|---|---|
| `val-core/allenai/IF_multi_constraints_upto5/reward/mean@8` | 동일 프롬프트당 8개 응답을 생성했을 때 평균 핵심 보상 |
| `val-core/allenai/IF_multi_constraints_upto5/reward/best@8/mean` | 8개 응답 중 최고 성능을 선택했을 때의 평균 잠재 성능 |
| `val-aux/allenai/IF_multi_constraints_upto5/reward_fmt/mean@8` | 8개 응답 기준 포맷 적합 평균 |
| `val-aux/allenai/IF_multi_constraints_upto5/reward_think/mean@8` | 8개 응답 기준 think/reasoning 구간 적합 평균 |
| `val-aux/allenai/IF_multi_constraints_upto5/score/mean@8` | scorer가 계산한 원점수 평균 |

## 7. 한눈에 보는 요약

| 구분 | 핵심 내용 |
|---|---|
| 데이터 성격 | instruction-following 평가용 RLVR/VERL 변환 데이터셋 |
| 활용 목적 | 제약 준수 여부 기반 자동 평가, RL post-training, instruction-following 성능 분석 |
| 핵심 장점 | 구조가 단순하고, 제약 유형이 다양하며, 자동 채점과 VERL 연동이 쉽다. |
| 해석 포인트 | scorer 원점수(`score`)와 veRL 핵심 보상(`reward`), 보조 지표(`reward_fmt`, `reward_think`)를 함께 봐야 원인 분석이 쉬워진다. |
