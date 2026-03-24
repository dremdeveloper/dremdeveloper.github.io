# DeepMath(DeepMath-103K) 데이터셋

## 1. 크기

| 항목 | 내용 |
|---|---|
| 공식 명칭 | DeepMath-103K |
| 총 샘플 수 | 103k rows |
| subset | default 1개 |
| split | train 단일 split |
| 저장 형식 | Parquet |
| 샤드 구성 | train-00000-of-00010 ~ train-00009-of-00010 |
| 데이터 용량 | data 디렉터리 약 2.14 GB, 리포지토리 전체 약 2.15 GB |
| 언어 | English |
| 라이선스 | MIT |

## 2. 형식

| 필드명 | 타입 | 설명 |
|---|---|---|
| `question` | `string` | 수학 문제 본문 |
| `final_answer` | `string` | 검증 가능한 최종 정답 |
| `difficulty` | `float64` | 난이도 점수 |
| `topic` | `string` | 계층형 주제 분류 |
| `r1_solution_1` | `string` | 장문 추론 풀이 1 |
| `r1_solution_2` | `string` | 장문 추론 풀이 2 |
| `r1_solution_3` | `string` | 장문 추론 풀이 3 |

> 한 샘플은 기본적으로 **문제 + 최종 정답 + 난이도 + 주제 + 3개 풀이** 구조로 구성된다.

## 3. rawdata 1개

```json
{
  "question": "Evaluate the limit: \\[ \\lim_{x \\to \\infty} \\sqrt{x} \\left( \\sqrt[3]{x+1} - \\sqrt[3]{x-1} \\right) \\]",
  "final_answer": "0",
  "difficulty": 4.5,
  "topic": "Mathematics -> Precalculus -> Limits",
  "r1_solution_1": "Okay, so I have this limit to evaluate...",
  "r1_solution_2": "Okay, so I need to evaluate the limit...",
  "r1_solution_3": "Okay, so I need to evaluate the limit..."
}
```

> 장문 `r1_solution_*` 필드는 길이가 길어 앞부분만 예시로 표기했다.

## 4. 특징

| 특징 | 설명 | 활용 포인트 |
|---|---|---|
| 고난도 중심 | 문제 난이도가 주로 Level 5~9에 집중되어 있다. | 고급 수학 추론 모델 학습에 적합 |
| 검증 가능한 정답 제공 | `final_answer`가 있어 규칙 기반 검증이 가능하다. | RL reward 설계와 자동 채점에 유리 |
| 주제 다양성 | Algebra, Calculus, Number Theory, Geometry, Probability, Discrete Mathematics 등 폭넓은 영역을 다룬다. | 범용 수학 추론 성능 점검 가능 |
| 엄격한 오염 제거 | 공통 benchmark와의 의미 기반 중복 제거를 거쳤다. | 평가 누수 감소 |
| 풍부한 레코드 구조 | 난이도, 주제, 정답, 3개 풀이를 한 레코드에 포함한다. | SFT, distillation, curriculum 학습에 활용 가능 |
| 단일 train 공개 구조 | 공개본은 train split 중심으로 제공된다. | 실사용 시 내부 validation/test 분리 필요 |

## 5. 한 줄 요약

DeepMath-103K는 **고난도 수학 문제**, **검증 가능한 최종 정답**, **난이도/주제 메타데이터**, **3개의 장문 풀이**를 함께 제공하는 대규모 수학 추론 학습 데이터셋이다.

## 6. 참고

- Hugging Face Dataset Card: DeepMath-103K
- GitHub README: zwhe99/DeepMath
- arXiv 2504.11456, *DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning*
