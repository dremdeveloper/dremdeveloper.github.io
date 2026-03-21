---
title: "LLM · RL 논문 용어집"
math: true
---

<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    tags: 'ams'
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  },
  svg: { fontCache: 'global' }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

# LLM · RL 논문 용어집

이 문서는 이번 묶음에 들어 있는 논문을 읽을 때 자주 나오는 용어를 한곳에 모아 정리한 것이다.  
대학 1학년이 처음 읽는다는 기준으로, 논문 본문에 자주 나오지만 처음 보면 막히기 쉬운 표현을 중심으로 골랐다.

## 1. 모델 구조와 표현 학습

| 용어 | 설명 |
|---|---|
| Attention | 입력의 각 위치가 다른 위치를 얼마나 참고할지 점수로 계산하는 방식이다. Transformer에서는 이 구조가 문장 안 관계를 잡는 핵심 장치다. |
| Self-Attention | 입력 문장 안의 각 토큰이 같은 문장 안 다른 토큰을 참고하는 attention이다. 예를 들어 문장 끝의 동사가 앞쪽 주어를 찾을 때 도움이 된다. |
| Query, Key, Value | attention을 계산할 때 쓰는 세 가지 벡터다. Query는 “무엇을 찾는가”, Key는 “무슨 정보를 갖고 있는가”, Value는 “실제로 전달할 정보는 무엇인가”에 가깝다. |
| Scaled Dot-Product Attention | query와 key의 내적(dot product)으로 유사도를 계산하고, 그 값을 $\sqrt{d_k}$로 나눈 뒤 softmax를 씌우는 attention 방식이다. 차원이 커질수록 점수가 너무 커지는 문제를 줄이기 위해 스케일링을 넣는다. |
| Multi-Head Attention | attention을 한 번만 하지 않고 여러 개의 head로 나누어 병렬로 계산하는 방식이다. 서로 다른 관계를 동시에 볼 수 있다는 점이 장점이다. |
| Positional Encoding | self-attention만 쓰면 순서 정보가 약해지기 때문에, 토큰 위치를 따로 알려 주는 장치가 필요하다. 이를 positional encoding이라고 한다. |
| Token | 모델이 문장을 처리할 때 쓰는 기본 단위다. 꼭 단어 하나일 필요는 없고, 단어 조각이나 기호 하나일 수도 있다. |
| Embedding | 토큰이나 문장을 숫자 벡터로 바꾼 표현이다. 비슷한 의미를 가진 입력은 벡터 공간에서도 가깝게 놓이는 경우가 많다. |
| Encoder | 입력 전체를 읽어 의미 표현을 만드는 블록이다. Transformer 원논문에서는 encoder와 decoder를 함께 쓴다. |
| Decoder | 이미 나온 출력과 encoder 정보를 참고해 다음 토큰을 생성하는 블록이다. LLM에서는 decoder-only 구조가 많이 쓰인다. |
| Feed-Forward Network | attention 뒤에 붙는 작은 MLP 블록이다. 각 위치별 표현을 한 번 더 비선형적으로 변환한다. |
| Residual Connection | 입력을 블록 출력에 더해 주는 연결이다. 깊은 신경망에서 학습이 불안정해지는 문제를 줄이는 데 도움을 준다. |
| Layer Normalization | 각 층의 출력 분포를 정규화해 학습을 안정화하는 기법이다. Transformer 계열에서 거의 기본처럼 쓰인다. |
| Softmax | 여러 점수를 확률처럼 보이도록 바꾸는 함수다. attention 가중치나 다음 토큰 확률을 만들 때 많이 쓴다. |
| Logit | softmax를 적용하기 전의 점수다. 모델이 각 토큰을 얼마나 선호하는지 원시 점수 형태로 담고 있다. |
| Context Length | 모델이 한 번에 볼 수 있는 최대 토큰 길이다. 문맥이 길어질수록 메모리와 계산량이 커진다. |
| Foundation Model | 매우 큰 범용 데이터로 먼저 학습한 기본 모델이다. 이후 여러 작업에 fine-tuning해서 쓴다. |
| Base Model | 정렬이나 대화 튜닝을 하기 전의 기본 언어모델을 가리키는 말이다. 보통 pretraining만 끝난 상태를 뜻한다. |
| Chat Model | 대화형 응답에 맞게 추가 학습한 모델이다. base model보다 말투, 안전성, 지시 따르기 능력이 보강되어 있다. |

## 2. 학습 방식과 최적화

| 용어 | 설명 |
|---|---|
| Pretraining | 거대한 일반 텍스트 데이터로 먼저 학습하는 단계다. 보통 다음 토큰 예측을 목표로 한다. |
| Next-Token Prediction | 앞 문맥이 주어졌을 때 다음 토큰을 맞히는 학습 목표다. 현재 LLM 사전학습의 가장 기본적인 목적식이다. |
| Fine-Tuning | 사전학습된 모델을 특정 목적에 맞게 다시 학습하는 과정이다. 도메인 적응, 대화 튜닝, 분류 작업 등에 모두 쓰인다. |
| Instruction Tuning | “질문/지시 → 응답” 형태의 데이터를 사용해 모델이 사람이 준 지시를 더 잘 따르도록 학습하는 방식이다. |
| Supervised Fine-Tuning (SFT) | 정답 응답이 있는 데이터로 지도학습하는 fine-tuning 단계다. RLHF 파이프라인의 첫 단계로 자주 쓰인다. |
| Alignment | 모델 출력이 사람이 원하는 기준과 더 잘 맞도록 만드는 작업 전체를 넓게 가리키는 말이다. |
| Loss Function | 모델이 얼마나 틀렸는지 숫자로 나타내는 함수다. 학습은 이 값을 줄이는 방향으로 진행된다. |
| Gradient | loss를 줄이기 위해 파라미터를 어느 방향으로 바꿔야 하는지 알려 주는 기울기다. |
| Learning Rate | 한 번 업데이트할 때 파라미터를 얼마나 크게 움직일지 정하는 값이다. 너무 크면 불안정하고 너무 작으면 학습이 느리다. |
| Warmup | 학습 초반에 learning rate를 천천히 올리는 기법이다. 대형 모델에서는 초반 발산을 막는 데 자주 쓴다. |
| Epoch | 전체 학습 데이터를 한 번 모두 사용하는 과정을 1 epoch라고 한다. |
| Batch | 한 번의 파라미터 업데이트에 함께 사용하는 데이터 묶음이다. |
| Label Smoothing | 정답 확률을 1로 두지 않고 조금 퍼뜨리는 기법이다. 모델이 지나치게 확신하는 것을 줄이는 데 도움을 준다. |
| Checkpoint | 학습 중간에 저장한 모델 상태다. 나중에 이어서 학습하거나 특정 시점 성능을 비교할 때 쓴다. |
| Transfer Learning | 한 작업에서 배운 표현을 다른 작업에 활용하는 학습 방식이다. LLM의 pretraining과 fine-tuning 구조가 대표적이다. |

## 3. 선호 학습과 정렬

| 용어 | 설명 |
|---|---|
| Human Feedback | 사람이 직접 “이 응답이 더 낫다” 혹은 “이 응답은 문제가 있다”고 평가해 주는 정보다. |
| RLHF | Reinforcement Learning from Human Feedback의 약자다. 사람 선호를 이용해 보상모델을 만들고, 그 보상을 이용해 정책을 강화학습하는 방식이다. |
| Preference Data | 같은 프롬프트에 대해 두 응답 중 어느 쪽이 더 나은지 비교한 데이터다. DPO나 RLHF에서 핵심 자원이다. |
| Reward Model | 응답이 얼마나 좋은지 점수로 예측하는 모델이다. 사람 비교 데이터를 보고 학습한다. |
| Reference Model | 정렬 과정에서 기준점으로 두는 모델이다. 보통 base model이나 SFT 모델을 그대로 두고, 새 정책이 너무 멀리 벗어나지 않게 하는 데 쓴다. |
| KL Divergence | 두 확률분포가 얼마나 다른지 재는 값이다. RLHF에서는 새 정책이 기존 모델에서 너무 멀어지는 것을 막는 제약으로 자주 쓴다. |
| DPO | Direct Preference Optimization의 약자다. reward model과 PPO를 따로 두지 않고, preference data로 정책을 직접 업데이트하는 방법이다. |
| Bradley–Terry Model | 두 후보를 비교할 때 어느 쪽이 선택될 확률을 점수 차이로 설명하는 간단한 비교 모델이다. DPO 설명에서 자주 등장한다. |
| Helpfulness | 응답이 실제로 도움이 되는가를 보는 기준이다. |
| Harmlessness | 응답이 해롭지 않은가를 보는 기준이다. 안전성 평가에서 중요하다. |
| Truthfulness | 응답이 사실과 얼마나 잘 맞는지를 보는 기준이다. 그럴듯하지만 틀린 답을 줄이는 문제와 연결된다. |
| Safety Tuning | 위험한 답변을 줄이고, 민감한 요청에 더 적절하게 대응하도록 추가 학습하는 과정이다. |
| Red Teaming | 공격적이거나 위험한 프롬프트를 던져 모델의 약점을 찾아내는 평가 방식이다. |
| Hallucination | 모델이 실제 근거 없이 그럴듯한 거짓 정보를 만들어 내는 현상이다. |
| Alignment Tax | 정렬을 강화하는 과정에서 원래의 일반 능력 일부가 떨어지는 비용을 가리키는 말이다. |

## 4. 강화학습 기본 수학

| 용어 | 설명 |
|---|---|
| Reinforcement Learning (RL) | 행동을 하고 그 결과로 보상을 받으면서 더 좋은 행동을 배우는 학습 방식이다. |
| Policy | 상태가 주어졌을 때 어떤 행동을 할지 정하는 규칙이다. 보통 $\pi(a\mid s)$처럼 쓴다. |
| State | 현재 상황을 요약한 정보다. 강화학습에서 에이전트는 상태를 보고 행동을 고른다. |
| Action | 에이전트가 실제로 취하는 선택이다. |
| Reward | 행동 뒤에 받는 즉시 점수다. 목표는 보통 장기적으로 reward 합을 크게 만드는 것이다. |
| Return | 한 시점 이후 받게 될 보상의 누적합이다. 보통 할인율을 넣어 $$ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots $$ 처럼 쓴다. |
| Discount Factor | 미래 보상을 얼마나 현재 가치로 줄여 볼지 정하는 값이다. 보통 $\gamma$로 쓴다. |
| Value Function | 어떤 상태나 행동이 장기적으로 얼마나 좋은지 기대값으로 나타낸 함수다. |
| Q-Value | 상태 $s$에서 행동 $a$를 했을 때 이후 기대 return을 나타내는 값이다. |
| Advantage | 그 행동이 평균보다 얼마나 더 좋은지 나타내는 값이다. 보통 $A(s,a)=Q(s,a)-V(s)$처럼 쓴다. |
| Baseline | 정책 경사 추정의 분산을 줄이기 위해 빼 주는 기준값이다. value function이 baseline 역할을 하는 경우가 많다. |
| Markov Property | 다음 상태와 보상 예측에 필요한 정보가 현재 상태 안에 충분히 들어 있다는 가정이다. |
| Transition Probability | 현재 상태와 행동이 주어졌을 때 다음 상태로 갈 확률이다. 보통 $p(s'\mid s,a)$처럼 쓴다. |
| One-Step Transition | 한 번 행동했을 때 상태가 어떻게 바뀌고 어떤 보상을 받는지 보는 가장 기본 단위다. |
| Bandit | 상태 전이가 없는 매우 단순한 강화학습 문제다. 한 번 선택하고 즉시 보상을 받는 구조만 본다. |
| Multi-Armed Bandit | 여러 슬롯머신 중 어떤 것을 당길지 배우는 문제로 비유하는 고전적 bandit 설정이다. |

## 5. PPO와 정책 최적화

| 용어 | 설명 |
|---|---|
| Policy Gradient | 정책 파라미터를 직접 미분해서 좋은 행동 확률을 높이는 방법이다. |
| PPO | Proximal Policy Optimization의 약자다. 정책을 너무 크게 바꾸지 않도록 clip을 넣어 안정성을 높인 방법이다. |
| Clipped Objective | 확률비가 너무 커지거나 작아질 때 업데이트 효과를 잘라 내는 목적식이다. PPO 핵심 아이디어다. |
| Probability Ratio | 새 정책과 옛 정책이 같은 행동에 부여한 확률의 비율이다. PPO에서는 이 값을 직접 사용한다. |
| Trust Region | 정책을 한 번에 너무 멀리 움직이지 않도록 제한하는 생각이다. PPO는 이를 근사적으로 구현한다. |
| On-Policy | 현재 학습 중인 정책이 직접 생성한 데이터로만 업데이트하는 방식이다. PPO가 여기에 속한다. |
| Sample Efficiency | 같은 성능을 내기 위해 얼마나 적은 데이터가 필요한지를 보는 개념이다. |

## 6. 검증 가능한 보상과 노이즈

| 용어 | 설명 |
|---|---|
| Verifiable Reward | 정답 여부를 외부 규칙이나 채점기로 비교적 명확하게 확인할 수 있는 보상이다. 코드 정답, 수학 채점 등이 예다. |
| Verifier | 응답이 맞았는지 틀렸는지 판정해 주는 검사기다. unit test, 규칙 기반 채점기, 별도 judge model 등이 될 수 있다. |
| RLVR | Reinforcement Learning with Verifiable Rewards의 약자다. 사람이 직접 점수를 주기보다 검증기 점수를 쓰는 방식이다. |
| False Positive | 사실은 틀렸는데 검증기가 맞았다고 판단하는 경우다. |
| False Negative | 사실은 맞았는데 검증기가 틀렸다고 판단하는 경우다. |
| TPR | True Positive Rate의 약자다. 실제 정답을 정답이라고 맞게 잡아내는 비율이다. |
| FPR | False Positive Rate의 약자다. 실제 오답을 정답이라고 잘못 통과시키는 비율이다. |
| Youden’s Index | 검증기 품질을 한 숫자로 보는 지표로 $$ J = \mathrm{TPR} - \mathrm{FPR} $$ 처럼 쓴다. $J>0$이면 랜덤보다 낫고, $J<0$이면 오히려 잘못된 방향으로 판단하는 경우가 많다. |
| Replicator Dynamics | 어떤 전략이 더 높은 보상을 받을수록 비율이 늘어나는 진화형 동역학이다. RLVεR 분석에서 학습 흐름을 설명할 때 등장한다. |
| Simplex | 확률들의 합이 1이 되는 공간이다. 여러 모드의 확률 질량이 어떻게 바뀌는지 분석할 때 쓴다. |
| Anti-Learning | 학습을 했는데 오히려 잘못된 행동이 더 강화되는 현상이다. noisy verifier가 너무 나쁘면 이런 문제가 생긴다. |

## 7. 평가와 데이터

| 용어 | 설명 |
|---|---|
| Benchmark | 모델을 비교하기 위해 만든 표준 평가셋이다. |
| BLEU | 기계번역 품질을 자동으로 재는 대표 지표다. 정답 문장과 얼마나 겹치는지 본다. |
| Human Evaluation | 실제 사람이 모델 출력을 보고 더 좋은 쪽을 고르거나 점수를 주는 평가다. |
| Unit Test | 코드가 특정 입력에서 올바른 출력을 내는지 검사하는 테스트다. 코드 RLVR에서 verifier 역할을 자주 한다. |
| Prompt Distribution | 실제 사용자가 던지는 질문들의 분포를 뜻한다. 학습용 프롬프트와 실제 사용 분포가 다르면 성능 체감도 달라질 수 있다. |
| Domain-Specific Instruction Tuning | 특정 분야 데이터와 지시 형식에 맞춰 instruction tuning을 수행하는 방식이다. 의료, 코드, 법률처럼 분야별 요구가 크게 다르다. |
| Zero-Shot | 예시를 따로 보여 주지 않고도 지시만으로 문제를 풀게 하는 설정이다. |
| Few-Shot | 문제를 풀기 전에 예시 몇 개를 함께 보여 주는 설정이다. |
| Chain-of-Thought | 정답만 바로 내지 않고 중간 추론 과정을 말로 적어 가는 방식이다. |
| Toxicity | 혐오, 공격, 유해 표현이 얼마나 많이 나오는지를 보는 개념이다. |
| Calibration | 모델의 자신감과 실제 정답 확률이 얼마나 잘 맞는지 보는 성질이다. |

## 8. 읽을 때 자주 헷갈리는 표현

| 용어 | 설명 |
|---|---|
| “모델이 크다” | 보통 파라미터 수가 많다는 뜻이다. 하지만 성능은 데이터 품질, 학습량, 정렬 방식에도 크게 좌우된다. |
| “오픈 모델” | 가중치나 사용 조건이 비교적 공개된 모델을 말한다. 완전 자유 사용을 뜻하는 것은 아니므로 라이선스를 따로 확인해야 한다. |
| “성능이 좋다” | 벤치마크 점수가 높다는 뜻일 수도 있고, 사람 평가에서 더 선호된다는 뜻일 수도 있다. 논문마다 기준이 다르니 무엇을 비교한 것인지 먼저 봐야 한다. |
| “안전하다” | 절대적으로 위험이 없다는 뜻이 아니라, 특정 평가와 공격 시나리오에서 상대적으로 더 나아졌다는 뜻으로 쓰이는 경우가 많다. |

