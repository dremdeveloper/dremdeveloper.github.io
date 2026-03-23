---
title: "AI 논문 용어집"
description: "articles/AI 논문 폴더의 논문 Markdown들을 바탕으로 정리한 기초 용어집"
---

# AI 논문 용어집

이 문서는 `articles/AI 논문/` 폴더에 있는 AI 논문 Markdown 파일들을 전수 확인한 뒤, **대학교 1학년이 처음 보면 낯설 수 있는 용어**를 골라 쉬운 말로 다시 정리한 용어집입니다.

- 조사 대상: DPO, Llama 2, PPO, RLHF, Attention Is All You Need, MoE 등 관련 Markdown 문서
- 제외 기준: 너무 일상적인 단어, 논문 내용과 직접 관련이 적은 단어
- 목표: 논문을 읽을 때 최소한의 맥락을 빠르게 잡도록 돕기

총 **65개 용어**를 정리했습니다.

---

## 용어집

| 번호 | 용어 | 쉬운 설명 |
| --- | --- | --- |
| 1 | Transformer | 문장 안에서 단어들끼리 서로 어떤 관련이 있는지 한꺼번에 살피는 대표적인 딥러닝 구조입니다. |
| 2 | Self-Attention | 문장 속 각 단어가 다른 단어를 얼마나 참고해야 하는지 계산하는 방법입니다. |
| 3 | Multi-Head Attention | 한 번만 보는 것이 아니라 여러 관점에서 동시에 attention을 계산해 더 다양한 관계를 잡는 방식입니다. |
| 4 | Positional Encoding | Transformer가 단어의 순서를 모르기 때문에, 각 단어의 위치 정보를 따로 넣어 주는 방법입니다. |
| 5 | Feedforward Network (FFN) | attention 계산 뒤에 각 토큰을 독립적으로 한 번 더 변환해 주는 신경망 층입니다. |
| 6 | Token | 모델이 문장을 처리할 때 사용하는 작은 단위입니다. 단어 전체일 수도 있고, 단어의 일부일 수도 있습니다. |
| 7 | Embedding | 단어나 토큰을 숫자 벡터로 바꿔서 모델이 계산할 수 있게 만든 표현입니다. |
| 8 | Language Model | 앞 문맥을 보고 다음 토큰이 무엇일지 예측하도록 학습된 모델입니다. |
| 9 | Large Language Model (LLM) | 아주 많은 데이터와 파라미터로 학습된 대규모 언어 모델입니다. |
| 10 | Pretraining | 대량의 일반 텍스트로 먼저 학습해 언어 감각과 기초 지식을 익히는 단계입니다. |
| 11 | Fine-Tuning | 이미 학습된 모델을 특정 목적에 맞게 추가로 학습시키는 단계입니다. |
| 12 | Supervised Fine-Tuning (SFT) | 질문과 정답처럼 정답이 있는 데이터로 모델을 미세조정하는 방식입니다. |
| 13 | Instruction Tuning | 사용자의 지시문을 잘 따르도록 여러 지시-응답 예시로 모델을 학습시키는 방법입니다. |
| 14 | Prompt | 모델에게 주는 입력 문장이나 지시문입니다. |
| 15 | Inference | 학습이 끝난 모델로 실제 예측이나 응답 생성을 수행하는 과정입니다. |
| 16 | Parameter | 모델 안에서 학습을 통해 값이 바뀌는 숫자들로, 모델의 지식을 담는 핵심 요소입니다. |
| 17 | Scaling Law | 데이터, 모델 크기, 연산량을 늘릴수록 성능이 어떻게 바뀌는지 설명하는 경험 법칙입니다. |
| 18 | Context Window | 모델이 한 번에 참고할 수 있는 입력 길이 범위입니다. |
| 19 | Autoregressive | 앞에서 생성한 토큰들을 바탕으로 다음 토큰을 하나씩 이어서 생성하는 방식입니다. |
| 20 | Zero-Shot | 별도 예시 없이 바로 문제를 풀게 하는 설정입니다. |
| 21 | Few-Shot | 문제를 풀기 전에 몇 개의 예시를 함께 보여 주는 설정입니다. |
| 22 | Alignment | 모델의 행동이 사람의 의도나 가치, 안전 기준에 더 잘 맞도록 조정하는 작업입니다. |
| 23 | RLHF | 사람의 선호를 이용해 언어 모델을 강화학습으로 정렬하는 방법으로, Reinforcement Learning from Human Feedback의 약자입니다. |
| 24 | RLAIF | 사람 대신 AI가 만든 피드백이나 평가를 활용해 모델을 정렬하는 방식입니다. |
| 25 | Preference Data | 두 응답 중 어느 쪽이 더 좋은지처럼 ‘선호’를 기록한 데이터입니다. |
| 26 | Preference Optimization | 선호 데이터에 맞게 모델이 더 좋은 응답을 내도록 학습하는 방법입니다. |
| 27 | DPO | 보상 모델과 별도 강화학습 없이 선호 데이터를 직접 이용해 정책을 학습하는 방법으로, Direct Preference Optimization의 약자입니다. |
| 28 | Reward Model | 어떤 응답이 더 좋은지 점수처럼 평가해 주는 모델입니다. |
| 29 | Policy | 현재 모델이 어떤 응답을 낼 확률분포를 뜻하며, 강화학습에서는 행동 규칙처럼 다룹니다. |
| 30 | Reference Model | 새 정책이 너무 멀리 벗어나지 않도록 비교 기준으로 두는 기존 모델입니다. |
| 31 | KL Divergence | 두 확률분포가 얼마나 다른지 재는 값입니다. 논문에서는 새 모델이 기준 모델에서 너무 멀어지는 것을 막는 데 자주 씁니다. |
| 32 | Regularization | 모델이 한쪽으로 과하게 치우치지 않도록 제약이나 벌점을 주는 기법입니다. |
| 33 | Bradley-Terry Model | 두 후보 중 어느 쪽이 더 선호될지 확률로 표현하는 비교 모델입니다. |
| 34 | Logistic Loss | 정답/오답처럼 두 선택지 중 하나를 맞히도록 학습할 때 자주 쓰는 손실 함수입니다. |
| 35 | Closed-Form Solution | 반복 계산 없이 식으로 바로 쓸 수 있는 해를 뜻합니다. |
| 36 | On-Policy | 현재 학습 중인 정책이 직접 만든 샘플을 사용해 다시 학습하는 방식입니다. |
| 37 | Off-Policy | 현재 정책이 아닌 과거 데이터나 다른 정책이 만든 데이터를 이용해 학습하는 방식입니다. |
| 38 | Reinforcement Learning (RL) | 행동을 했을 때 받는 보상을 최대화하도록 학습하는 방법입니다. |
| 39 | PPO | 정책을 너무 급하게 바꾸지 않으면서 강화학습을 안정적으로 진행하는 알고리즘으로, Proximal Policy Optimization의 약자입니다. |
| 40 | Policy Gradient | 정책이 더 높은 보상을 받도록 확률을 직접 조정하는 강화학습 방법입니다. |
| 41 | Value Function | 지금 상태가 앞으로 얼마나 좋은 보상을 기대할 수 있는지 추정하는 함수입니다. |
| 42 | Baseline | 강화학습에서 보상 변동을 줄여 학습을 안정화하기 위해 빼 주는 기준값입니다. |
| 43 | Advantage | 실제로 받은 결과가 기준 기대치보다 얼마나 더 좋았는지를 나타내는 값입니다. |
| 44 | Rejection Sampling | 여러 후보를 만든 뒤 기준에 맞지 않는 것을 버리고 좋은 것만 고르는 방식입니다. |
| 45 | Rejection Sampling Fine-Tuning | 여러 응답 중 더 나은 응답만 골라서 미세조정 데이터로 쓰는 방법입니다. |
| 46 | Gating Network | MoE에서 어떤 expert를 사용할지 고르는 작은 선택 네트워크입니다. |
| 47 | Mixture of Experts (MoE) | 여러 전문가 모듈 중 일부만 골라 계산하게 해서 큰 모델을 더 효율적으로 쓰는 구조입니다. |
| 48 | Expert | MoE 안에서 특정 입력 패턴을 더 잘 처리하도록 분업하는 하위 모듈입니다. |
| 49 | Sparse Activation | 전체 모듈을 다 쓰지 않고 일부만 활성화해서 계산량을 줄이는 방식입니다. |
| 50 | Routing | 입력을 어떤 expert에게 보낼지 결정하는 과정입니다. |
| 51 | Top-k Routing | 점수가 높은 expert 중 상위 k개만 선택해 사용하는 라우팅 방식입니다. |
| 52 | Load Balancing | 일부 expert에만 일이 몰리지 않도록 입력을 비교적 고르게 분배하는 장치입니다. |
| 53 | Expert Collapse | 몇몇 expert만 계속 선택되고 나머지는 거의 쓰이지 않는 문제입니다. |
| 54 | Token Dropping | expert 수용 한계를 넘은 토큰이 라우팅 과정에서 버려지는 현상입니다. |
| 55 | Sharding | 큰 모델이나 데이터를 여러 장치에 나눠 저장하고 계산하는 방법입니다. |
| 56 | Parallelism | 여러 연산을 동시에 수행해 학습이나 추론 속도를 높이는 방식입니다. |
| 57 | Quantization | 숫자 표현 정밀도를 낮춰 메모리와 계산량을 줄이는 기술입니다. |
| 58 | Low-Rank Adapter (LoRA) | 기존 큰 모델 전체를 바꾸지 않고 작은 추가 모듈만 학습해 효율적으로 튜닝하는 방법입니다. |
| 59 | Distillation | 큰 모델이나 여러 모델의 지식을 더 작거나 다른 모델에 옮겨 주는 학습 방식입니다. |
| 60 | Generalization | 학습 때 보지 못한 새로운 데이터에서도 잘 작동하는 능력입니다. |
| 61 | Hallucination | 모델이 그럴듯하지만 사실이 아닌 내용을 자신 있게 만들어 내는 현상입니다. |
| 62 | Red Teaming | 모델의 취약점이나 위험한 응답을 의도적으로 찾아보는 점검 과정입니다. |
| 63 | Safety Tuning | 위험한 출력은 줄이고 더 안전한 응답을 하도록 추가 학습하는 작업입니다. |
| 64 | Benchmark | 여러 모델을 같은 기준으로 비교하기 위한 평가용 데이터셋이나 시험 체계입니다. |
| 65 | Robustness | 입력이 조금 바뀌거나 까다로운 상황에서도 성능이 쉽게 무너지지 않는 성질입니다. |

---

## 읽는 순서 추천

처음 읽는 사람이라면 아래 순서로 이해하면 부담이 덜합니다.

1. **Transformer, Token, Embedding, Self-Attention**부터 이해하기
2. 그다음 **Pretraining, Fine-Tuning, Instruction Tuning** 보기
3. 이후 **Alignment, RLHF, Reward Model, PPO, DPO**로 넘어가기
4. 마지막으로 **MoE, Expert, Routing, Load Balancing** 같은 대규모 모델 구조 읽기
