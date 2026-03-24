# DPO · PPO · GRPO 비교

> 목적: 이 문서는 **각 논문을 개별적으로 해설**하는 문서가 아니라, **LLM 후학습(post-training) / 정렬(alignment) 관점에서 DPO, PPO, GRPO를 같은 축 위에 올려놓고 비교**하기 위해 작성했다.  
> 전제: 본 문서의 PPO 설명은 **일반 강화학습 전체**보다 좁은 범위인 **LLM RLHF 맥락에서 쓰이는 PPO**를 기준으로 한다. 따라서 여기서 말하는 reward model, reference model, value model은 **LLM 정렬 파이프라인에서의 PPO**에 대한 설명이다.
> 근거 자료: **DPO 원 논문, PPO 원 논문, DeepSeekMath(GRPO) 논문, InstructGPT, OpenAI Spinning Up PPO, Hugging Face TRL의 DPO/PPO/GRPO 문서**를 교차 참조했다.

---

## 0. 먼저 결론: 세 방법은 "같은 목표를 다른 방식으로 푼다"

세 알고리즘은 모두 결국 다음 계열의 문제를 다른 방식으로 다룬다고 볼 수 있다.

$$
\max_{\pi} \; \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi(\cdot|x)}[r(x,y)] 
\;-
\beta D_{KL}(\pi(\cdot|x)\,\|\,\pi_{ref}(\cdot|x))
$$

즉,

- **좋은 응답에는 더 높은 점수**를 주고,
- 동시에 **기준(reference) 모델에서 너무 멀어지지 않게** 만들고,
- 이 과정을 통해 **사람이 선호하는 방향으로 정책(policy)을 이동**시키려는 점은 세 방법이 공유한다.

하지만 그 목표를 푸는 방식은 크게 다르다.

- **PPO**는 이 문제를 **온라인 RL(on-policy RL)** 로 푼다.
- **DPO**는 같은 목표를 **선호쌍(pairwise preference)을 직접 분류하는 목적함수**로 바꿔 푼다.
- **GRPO**는 PPO 계열을 유지하되, **critic/value model을 제거**하고 **같은 프롬프트에서 샘플한 여러 응답의 상대 점수**로 advantage를 만든다.

이 차이 때문에 세 방법은 **필요한 데이터**, **필요한 모델 구성**, **메모리/토큰 비용**, **안정성**, **잘 맞는 과제 종류**가 달라진다.

---

## 1. 한 장으로 보는 핵심 비교

| 비교 축 | DPO | PPO | GRPO |
|---|---|---|---|
| 본질 | 선호쌍을 직접 최적화하는 **오프라인 preference optimization** | reward를 최대화하는 **온라인 RL** | PPO 계열의 **온라인 RL**이지만 value model 없이 **group-relative advantage** 사용 |
| 학습 입력 | `prompt + chosen + rejected` | `prompt + rollout + reward` | `prompt + prompt당 여러 rollout + reward` |
| 정책 학습 단계의 명시적 reward model | **불필요** | 보통 **필요** (LLM RLHF) | **필요하거나**, reward function 사용 |
| value / critic | 불필요 | **필요** | **불필요** |
| reference model | 중요함 (목적함수 안에 직접 들어감) | LLM RLHF에서는 보통 KL 제어용으로 사용 | KL 제어용으로 사용 |
| old policy 의존성 | 없음 | 큼 | 큼 |
| 데이터 성격 | 정적인 선호 데이터셋 | 온정책(on-policy) 샘플 | 온정책(on-policy) 샘플 + 그룹 비교 |
| 업데이트 단위 | pairwise preference sample | trajectory / token / minibatch | same-prompt grouped completions |
| 장점 | 단순, 가벼움, 비교적 안정적, 구현 쉬움 | 가장 일반적이고 유연함, 온라인 탐색 가능 | PPO보다 메모리 부담 작고 reasoning 과제에 잘 맞는 편 |
| 단점 | 온라인 탐색 불가, reward shaping/step reward를 직접 쓰기 어려움 | reward/value/KL 등 구성요소가 많아 무겁고 튜닝 난이도 높음 | value는 없지만 group sampling 비용이 있고, group size 및 reward 분산에 민감 |
| 잘 맞는 상황 | 고정된 선호쌍 데이터로 정렬하고 싶을 때 | reward model / verifier / 환경 상호작용을 직접 최적화해야 할 때 | reasoning·math·code처럼 **같은 문제에 여러 샘플을 비교**하는 것이 자연스러운 과제 |

> **중요한 한 줄 요약**  
> - **DPO**: “선호쌍이 이미 있다면, RL 없이 직접 최적화하자.”  
> - **PPO**: “보상을 최대화하는 정책을 온라인 RL로 점진적으로 밀어 올리자.”  
> - **GRPO**: “온라인 RL은 유지하되, critic 없이 같은 프롬프트의 여러 응답을 서로 비교해 advantage를 만들자.”

> **주의**: 위 비교는 어디까지나 **LLM 정렬 맥락**의 PPO/GRPO 기준이다. 일반 RL 논문으로서의 PPO는 reward model이나 reference model을 전제하지 않는다. 다만 LLM RLHF 구현에서는 보통 이들이 함께 등장한다.

---

## 2. 왜 이 셋이 같이 비교되는가

이 셋이 자주 함께 언급되는 이유는, 세 방법이 **“사람 선호 또는 외부 보상에 맞춰 LLM을 후학습하는 방법”**이라는 같은 문제군에 속하기 때문이다.

1. **PPO**는 RLHF의 대표적 고전 구현이다.  
   InstructGPT 계열 파이프라인은 대체로 **(a) 시연 데이터로 SFT → (b) 비교 데이터로 reward model 학습 → (c) PPO로 정책 최적화**의 구조를 취한다.

2. **DPO**는 “굳이 reward model을 따로 학습하고 PPO까지 돌리지 않아도, 선호쌍이 있으면 더 직접적으로 최적화할 수 있지 않나?”라는 문제의식에서 나왔다.

3. **GRPO**는 PPO의 온라인 RL 장점을 살리되, 특히 LLM 환경에서 부담이 큰 **value model**을 제거하고, **same-prompt multi-sample 비교**를 이용해 더 가볍고 reasoning 친화적인 학습을 하려는 방향이다.

즉, 비교의 축은 이렇게 정리할 수 있다.

- **PPO ↔ DPO**: RLHF를 RL로 할 것인가, 선호최적화로 직접 풀 것인가?
- **PPO ↔ GRPO**: 온라인 RL을 유지할 때 critic/value model이 꼭 필요한가?
- **DPO ↔ GRPO**: 둘 다 “상대 비교”를 활용하지만, 하나는 **오프라인 pairwise preference**, 다른 하나는 **온라인 group-relative advantage**라는 점에서 본질적으로 다르다.

### 이 비교에서 가장 중요한 오해 방지 포인트

**DPO와 GRPO는 둘 다 “상대성(relative comparison)”을 쓰지만, 상대성의 위치가 다르다.**

- **DPO의 상대성**: 데이터셋에 들어있는 **chosen vs rejected 쌍**에 있다.
- **GRPO의 상대성**: 학습 중 같은 프롬프트에서 생성한 **여러 출력들 사이의 group-relative reward**에 있다.

즉,

- DPO는 **오프라인 pairwise preference likelihood** 이고,
- GRPO는 **온라인 policy gradient with grouped relative advantages** 이다.

둘을 같은 “비교 학습”이라고만 보면 본질을 놓치게 된다.

---

## 3. 수식 관점에서 보면 무엇이 다른가

이 섹션이 가장 중요하다. 세 방법의 차이는 결국 **무엇을 분모로 두고**, **무슨 신호로 policy를 밀며**, **KL 제약을 어디에 넣는가**에서 드러난다.

### 3.1 PPO: 이전 정책(old policy) 대비 얼마나 움직였는가를 제어한다

PPO의 핵심 식은 다음이다.

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

$$
L^{CLIP}(\theta)=\hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t\big)\Big]
$$

핵심은 다음이다.

- PPO의 **직접적인 분모**는 보통 **reference model이 아니라 old policy**다.
- 즉 PPO의 핵심 관심사는 “이번 업데이트가 **직전 정책에서 너무 크게 벗어나지 않게** 하면서 성능을 올리자”이다.
- 여기에 LLM RLHF 맥락이 들어오면, 보상 설계에서 추가로 **reference model 대비 KL 패널티**를 넣는다.

LLM RLHF에서는 보통 다음과 같은 구조가 된다.

- 현재 정책이 응답을 생성한다.
- reward model이 그 응답에 점수를 준다.
- reference model과의 KL을 패널티로 넣는다.
- value model이 advantage 추정을 돕는다.
- PPO clip objective로 policy를 업데이트한다.

즉 PPO는 **정책 업데이트 안정성**은 `old policy`에 대한 ratio clipping으로 관리하고, **모델 붕괴/분포 이탈 방지**는 `reference model`과의 KL로 관리하는 경우가 많다.

#### PPO의 강점

- 온라인 RL이므로 **현재 정책이 실제로 생성한 샘플**을 바탕으로 개선할 수 있다.
- reward model이나 programmatic reward, verifier reward 등 **스칼라 보상**을 직접 사용할 수 있다.
- dense reward, sparse reward, token-level reward, sequence-level reward 등 비교적 일반적인 틀을 제공한다.

#### PPO의 약점

- value model이 필요해 메모리/연산 부담이 크다.
- reward model, KL coefficient, clip range, value loss coefficient 등 **튜닝 포인트가 많다**.
- 잘못 설계된 reward나 과한 최적화로 **reward hacking**이나 분포 붕괴가 생길 수 있다.

---

### 3.2 DPO: reward model을 명시적으로 학습하지 않고, 선호쌍을 직접 최적화한다

DPO의 핵심은, KL-제약된 RLHF objective의 최적 정책을 이용해 **reward model을 명시적으로 거치지 않고 policy를 직접 학습**할 수 있다는 점이다.

DPO의 대표 식은 다음과 같다.

$$
\mathcal{L}_{DPO}(\pi_\theta;\pi_{ref})
=
-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\left[
\log \sigma\left(
\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right)
\right]
$$

여기서 핵심은 다음이다.

- DPO의 직접적인 비교 대상은 **old policy가 아니라 reference model**이다.
- 그리고 비교 단위는 trajectory가 아니라 **chosen / rejected 쌍**이다.
- 다시 말해 DPO는 “현재 정책이 reference에 비해 **선호된 응답에는 더 높은 상대 로그확률**, 비선호 응답에는 더 낮은 상대 로그확률을 주도록” 학습한다.

이 점 때문에 DPO는 PPO와 완전히 다른 느낌을 준다.

- PPO는 **rollout → reward → advantage → policy gradient**라는 RL 루프를 가진다.
- DPO는 **pairwise preference → logistic loss**라는 **분류형 최적화**에 더 가깝다.

#### DPO가 단순한 이유

정책 학습 단계에서 DPO는 보통 다음만 있으면 된다.

- 현재 정책 모델
- frozen reference model
- preference dataset (`prompt`, `chosen`, `rejected`)

즉 보통 **reward model도, value model도 필요 없다.**

#### 그렇다고 DPO가 “보상 개념이 전혀 없는 것”은 아니다

이 표현은 절반만 맞다.

- 맞는 부분: **정책 학습 단계에서 명시적 reward model을 따로 학습하지 않는다.**
- 틀리기 쉬운 부분: DPO는 여전히 **선호를 만족하는 암묵적 reward 구조**를 이론적으로 전제한다.

따라서 DPO는 “reward 없는 방법”이라기보다,

> **명시적 reward model fitting을 생략하고, 선호쌍에서 policy를 직접 학습하는 방법**

이라고 이해하는 편이 정확하다.

#### DPO의 강점

- RL loop가 없어서 구현이 단순하다.
- 온정책 샘플링이 없으므로 토큰 비용과 시스템 복잡도가 낮다.
- 보통 PPO 기반 RLHF보다 실험 안정성이 높고 재현성이 좋은 편이다.

#### DPO의 약점

- 온라인 탐색이 없다. 즉 **학습 중 새로 생성한 응답을 탐험하면서 개선**하는 구조가 아니다.
- reward function이나 verifier signal이 있어도, 그것을 직접 쓰기보다는 보통 **pairwise preference 형태로 변환**해야 한다.
- step-wise/process reward를 자연스럽게 쓰기 어렵다.

---

### 3.3 GRPO: PPO의 온라인 RL 구조는 유지하되, value model을 버리고 group-relative advantage를 쓴다

GRPO는 DeepSeekMath에서 제안된 PPO 변형이다. 핵심 아이디어는 간단하다.

> **같은 프롬프트에 대해 여러 응답을 샘플링하고, 그들 사이의 상대 점수로 baseline/advantage를 만들면, critic(value model)을 따로 학습하지 않아도 된다.**

즉 GRPO는 PPO처럼 정책을 온라인으로 업데이트하지만, advantage를 만들기 위해 **학습된 value function**에 덜 의존하거나 아예 제거한다.

논문의 설명을 outcome supervision 관점에서 단순화하면,

1. 질문 `q`에 대해 현재(정확히는 old policy) 정책에서 여러 응답 `o_1, ..., o_G`를 샘플링한다.
2. 각 응답에 reward `r_1, ..., r_G`를 부여한다.
3. 이 reward를 같은 그룹 안에서 정규화한다.

$$
\tilde r_i = \frac{r_i - \text{mean}(\mathbf r)}{\text{std}(\mathbf r)}
$$

4. 이 그룹 상대 점수를 advantage처럼 사용해 PPO 유사 objective를 최적화한다.

즉 GRPO의 핵심은 다음 두 문장으로 압축된다.

- **PPO처럼 온라인 RL**을 한다.
- **critic 대신 same-prompt group comparison**을 쓴다.

#### GRPO의 objective가 PPO와 닮았지만 같지는 않은 이유

GRPO는 PPO와 마찬가지로 clip을 사용하지만, advantage 추정 방식이 다르다.

- PPO: value model을 기반으로 한 advantage/GAE
- GRPO: group reward의 평균/표준편차를 기반으로 한 relative advantage

또한 DeepSeekMath 논문은, PPO 스타일 RLHF에서 흔히 reward 안에 KL 패널티를 넣는 것과 달리, **GRPO에서는 KL divergence를 loss에 직접 더하는 방식**을 강조한다. 이 차이는 중요하다.

즉 GRPO는 단순히 “critic 없는 PPO”가 아니라,

- **same-prompt multi-sample**이 본질이고,
- **group-relative normalization**이 핵심이며,
- KL regularization을 objective에 직접 넣는 설계를 강조하는 점이 특징이다.

#### GRPO의 강점

- PPO보다 메모리 부담이 줄어든다. 특히 value model이 빠진다는 점이 크다.
- 같은 문제에 대해 여러 샘플의 우열을 비교하는 구조라, **reasoning, math, code** 같은 과제에 잘 맞는다.
- reward model 외에도 정답 검증기, unit test, 형식 검사기, 실행기 등 **검증 가능한 외부 reward**와 결합하기 좋다.

#### GRPO의 약점

- DPO처럼 가벼운 것은 아니다. 여전히 **온라인 샘플링 비용**이 든다.
- prompt마다 여러 응답을 뽑아야 하므로, group size `G`가 커지면 토큰 비용이 빠르게 늘어난다.
- 그룹 내 reward 분산이 너무 작으면 학습 신호가 약해질 수 있다.
- group normalization, KL 설정, 길이 편향(length bias) 문제 등 구현 세부가 성능에 영향을 준다.

---

## 4. 세 방법을 가장 잘 갈라놓는 기준: "학습 신호를 어디서 받는가"

이 비교가 실무적으로 가장 중요하다.

### DPO의 학습 신호

DPO는 **이미 주어진 선호쌍**에서 학습 신호를 받는다.

- 이 응답이 저 응답보다 낫다.
- chosen이 rejected보다 선호된다.
- 현재 정책이 reference 대비 chosen을 얼마나 더 밀고 rejected를 얼마나 덜 미는가.

즉 학습 신호의 원천이 **데이터셋 내부**에 있다.

### PPO의 학습 신호

PPO는 **현재 정책이 실제로 생성한 샘플**에서 학습 신호를 받는다.

- 정책이 응답을 만든다.
- reward model / reward function이 점수를 준다.
- value model이 baseline을 제공한다.
- advantage를 통해 policy gradient를 만든다.

즉 학습 신호의 원천이 **학습 중 생성된 rollout + 보상 함수**에 있다.

### GRPO의 학습 신호

GRPO도 생성된 샘플에서 신호를 받지만, 포인트는 **개별 샘플의 절대 점수보다 그룹 내 상대 위치**를 중요하게 본다는 점이다.

- 같은 프롬프트에서 여러 응답을 생성한다.
- 각 응답의 reward를 구한다.
- 그 reward를 그룹 내에서 비교/정규화한다.
- 이 상대성을 advantage로 쓴다.

즉 GRPO의 학습 신호는 **“온라인으로 생성된 여러 응답들 사이의 상대 비교”** 에서 나온다.

### 핵심 정리

- **DPO**: 정적 데이터셋 기반의 상대 비교  
- **PPO**: 온라인 rollout 기반의 절대/누적 보상 최적화  
- **GRPO**: 온라인 rollout 기반이지만 same-prompt group 내부의 상대 비교를 강조

이 차이가 곧 **오프라인 vs 온라인**, **pairwise vs grouped**, **critic 필요 여부** 차이로 이어진다.

---

## 5. 학습 루프 자체는 어떻게 다른가

### 5.1 DPO 학습 루프

1. SFT 또는 base model을 시작점으로 잡는다.
2. reference model을 고정한다.
3. preference dataset에서 `(prompt, chosen, rejected)` 배치를 읽는다.
4. 현재 정책과 reference가 chosen/rejected에 부여하는 log-prob를 계산한다.
5. DPO loss를 계산해 정책을 업데이트한다.

즉 DPO는 본질적으로 **“pairwise classification-style finetuning loop”** 다.

### 5.2 PPO 학습 루프

1. 현재 정책으로 프롬프트에 대한 응답을 생성한다.
2. reward model / reward function으로 보상을 계산한다.
3. reference model과의 KL을 보상 또는 auxiliary term으로 반영한다.
4. value model로 advantage 또는 return baseline을 추정한다.
5. 같은 on-policy batch에 대해 여러 epoch 동안 PPO clip objective로 업데이트한다.

즉 PPO는 **생성-평가-가치추정-정책업데이트**가 한 덩어리로 묶인 RL loop다.

### 5.3 GRPO 학습 루프

1. 프롬프트를 샘플한다.
2. 각 프롬프트마다 여러 응답을 생성한다.
3. 각 응답에 reward를 계산한다.
4. 그룹 내부에서 reward를 정규화해 relative advantage를 만든다.
5. PPO류 clip objective로 업데이트한다.

즉 GRPO는 **생성-그룹비교-정책업데이트** 구조다. PPO보다 value estimation 단계가 간소화된다.

### 실무 감각으로 요약하면

- DPO는 **dataset-centric**  
- PPO는 **rollout-centric**  
- GRPO는 **grouped-rollout-centric**

라고 볼 수 있다.

---

## 6. 어떤 모델/구성요소가 필요한가

이 섹션은 실제 학습 비용을 가르는 핵심이다.

| 구성요소 | DPO | PPO | GRPO |
|---|---|---|---|
| 학습 대상 policy model | 필요 | 필요 | 필요 |
| frozen reference model | 필요 | 보통 필요 | 보통 필요 |
| reward model / reward function | 정책 학습 단계에서는 보통 불필요 | 필요 | 필요 |
| value model / critic | 불필요 | 필요 | 불필요 |
| on-policy generation | 불필요 | 필요 | 필요 |
| prompt당 다중 샘플 | 필수 아님 | 필수 아님 | 사실상 핵심 |

### DPO가 가장 가벼운 이유

DPO는 정책 학습 단계에서 보통

- reward model이 없고,
- value model이 없고,
- rollout generation도 없다.

그래서 **시스템 구조가 가장 단순**하다.

### PPO가 가장 무거운 이유

PPO는 LLM RLHF에서 흔히 다음을 동시에 다룬다.

- policy model
- reference model
- reward model
- value model
- on-policy generation 파이프라인

즉 **모델 수도 많고, 샘플링도 필요하고, 업데이트 로직도 복잡**하다.

### GRPO는 PPO보다 가벼우나, DPO보다 가볍지는 않다

GRPO는 value model을 제거하므로 PPO보다 가벼워지는 경향이 있다. 하지만 DPO와 비교하면 여전히

- 온라인 generation이 필요하고,
- prompt마다 여러 샘플을 생성해야 하며,
- reward scoring을 계속 해야 한다.

따라서 비용 관점에서 대체로 다음처럼 생각하면 된다.

**대체로**

$$
\text{DPO (가장 가벼움)} \;<\; \text{GRPO} \;<\; \text{PPO (가장 무거운 편)}
$$

단, GRPO의 group size가 크고 completion 길이가 길면, 실제 토큰 비용은 상당히 커질 수 있다.

---

## 7. "분모가 누구인가"로 보면 더 명확해진다

세 방법의 차이를 아주 빠르게 파악하는 좋은 방법은 **각 objective에서 확률비의 분모가 누구인지** 보는 것이다.

### PPO

$$
\frac{\pi_\theta}{\pi_{old}}
$$

- 관심사: **업데이트 안정성**
- 의미: “이전 정책에 비해 이번 정책이 얼마나 달라졌는가?”

### DPO

$$
\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)},\quad
\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
$$

- 관심사: **reference 대비 chosen과 rejected의 상대적 마진**
- 의미: “reference보다 chosen을 더 올리고 rejected를 더 내렸는가?”

### GRPO

- clipping 쪽 ratio는 보통 **current vs old policy**
- regularization 쪽은 **current policy vs reference model**

즉 GRPO는 사실상

- **PPO의 업데이트 안정성 구조**와
- **RLHF의 reference regularization 구조**를

같이 갖고 있는 셈이다.

이 관점에서 보면 세 방법의 차이가 아주 선명해진다.

- PPO: `old policy` 중심
- DPO: `reference model` 중심
- GRPO: `old policy + reference model` 이중 구조

---

## 8. 데이터 요건과 과제 적합성은 어떻게 다른가

### 8.1 DPO가 잘 맞는 데이터

DPO는 다음과 같은 데이터에 잘 맞는다.

- 인간 선호 비교 데이터
- chosen/rejected 형식의 synthetic preference 데이터
- 기존 reward나 verifier를 pairwise preference로 변환한 데이터

즉 **쌍 비교 데이터가 이미 있을 때** 특히 좋다.

### 8.2 PPO가 잘 맞는 데이터 / 보상

PPO는 다음과 같은 경우에 자연스럽다.

- reward model이 따로 있을 때
- 시뮬레이터나 환경 상호작용이 있을 때
- scalar reward를 직접 최적화하고 싶을 때
- token-level 또는 sequence-level reward가 있을 때

즉 **pairwise dataset보다 “보상 함수”가 중심**일 때 적합하다.

### 8.3 GRPO가 잘 맞는 데이터 / 보상

GRPO는 특히 같은 프롬프트에 대해 여러 답안을 뽑고 그 사이를 비교하는 것이 자연스러운 작업에 잘 맞는다.

예를 들면,

- 수학 문제 풀이
- 코드 생성 및 테스트 통과 여부
- 정답 검증이 가능한 reasoning task
- 형식/구문/검증기를 붙이기 쉬운 과제

GRPO가 reasoning 과제에 잘 맞는 이유는,

- 같은 문제에 대해 여러 해법을 샘플링할 수 있고,
- 그 해법들을 reward 기준으로 비교해 상대 advantage를 만들 수 있으며,
- value model 없이도 그룹 상대성만으로 학습 신호를 만들 수 있기 때문이다.

### 중요한 실무적 차이

- **DPO**는 보통 “이미 있는 선호 데이터”를 최대한 잘 활용하는 쪽에 강하다.
- **PPO/GRPO**는 “학습 중에 새로 생성한 응답”을 바탕으로 개선하는 쪽에 강하다.

즉,

- **정적 정렬**에 가까우면 DPO가 유리하고,
- **동적 탐색**이 중요하면 PPO/GRPO가 유리하다.

---

## 9. advantage를 어떻게 얻느냐가 성격을 결정한다

### PPO: learned value baseline

PPO의 advantage는 보통 **value function**에서 온다.

- 현재 상태/토큰 prefix에서 기대되는 가치를 추정하고,
- 실제 return 또는 reward와의 차이로 advantage를 만든다.
- LLM RLHF에서는 흔히 GAE나 그 변형을 사용한다.

이 방식의 장점은 일반성이 높다는 점이지만, LLM처럼 길고 고차원적인 생성에서는 **좋은 value function을 안정적으로 학습하는 것 자체가 어렵고 비싸다**.

### DPO: advantage 개념 자체가 다르다

DPO에는 PPO식 advantage 추정이 없다.

대신 다음 비교가 핵심이다.

- chosen이 rejected보다 얼마나 더 선호되는가
- 그리고 현재 정책이 reference 대비 이 차이를 얼마나 더 벌렸는가

즉 DPO는 **policy gradient + advantage** 구조가 아니라, **pairwise preference likelihood** 구조다.

### GRPO: group-relative baseline

GRPO는 critic을 학습하지 않고,

- 같은 질문에 대한 여러 출력의 reward를 비교하고,
- 그 평균과 표준편차로 정규화해,
- 상대적으로 더 좋은 출력에 양의 신호, 더 나쁜 출력에 음의 신호를 준다.

이 점 때문에 GRPO는 아주 본질적으로 **“ranking-aware online RL”** 같은 성격을 띤다.

### 그래서 무엇이 달라지나

- PPO는 **절대적 가치 추정**에 기대고,
- DPO는 **정적 pairwise preference**에 기대고,
- GRPO는 **동적 group-relative comparison**에 기대는 셈이다.

---

## 10. KL regularization을 어디에 넣는가도 다르다

이 차이는 논문을 읽어도 종종 놓치기 쉬운 포인트다.

### PPO 계열 RLHF

LLM RLHF에서 PPO는 흔히 reward에 KL 패널티를 합친다.

직관적으로는,

- reward model 점수를 높이고 싶지만,
- reference에서 너무 멀어지면 패널티를 준다.

즉 KL이 **보상 쪽**으로 들어간다.

### DPO

DPO에서는 reference model이 loss 안에 직접 들어간다.  
다시 말해 KL 패널티를 별도 reward shaping처럼 더하는 느낌보다,

- chosen/rejected의 **reference 대비 상대 로그확률 마진**을 직접 학습하는 구조다.

### GRPO

DeepSeekMath의 설명 포인트는, PPO형 RLHF처럼 reward에 KL을 얹는 대신, **GRPO에서는 KL divergence를 loss 쪽에 직접 두는 설계**를 강조한다는 점이다.

이 차이는 실무에서 중요하다.

- reward shaping 위치가 달라지면 advantage 계산의 의미도 달라지고,
- KL coefficient가 학습 동역학에 미치는 영향도 달라진다.

즉 세 방법은 모두 KL을 중요하게 다루지만, **KL이 objective의 어느 위치에서 작동하는지**가 서로 다르다.

---

## 11. 안정성, 튜닝 난이도, 재현성 비교

### 11.1 DPO

상대적으로 단순하고 안정적인 편이다.

- reward model 학습 불필요
- value model 불필요
- rollout 불필요
- 선호쌍만 있으면 바로 최적화 가능

하지만 DPO도 하이퍼파라미터에 완전히 둔감한 것은 아니다.

- `β` 값이 너무 크면 reference에서 과하게 이탈하거나 선호 마진을 과도하게 밀 수 있다.
- 너무 작으면 선호신호를 충분히 반영하지 못할 수 있다.
- 데이터 품질이 낮으면 chosen/rejected의 노이즈를 그대로 학습한다.

즉 DPO는 **단순하지만, 데이터 품질과 β 설정에 민감**하다.

### 11.2 PPO

세 방법 중 튜닝 난이도가 가장 높은 편이다.

주요 민감 포인트는 다음과 같다.

- clip range `ε`
- KL coefficient
- reward scale / normalization
- value loss coefficient
- minibatch 수와 PPO epoch 수
- sampling temperature / rollout length

또한 LLM RLHF에서는 reward model이 imperfect하기 때문에, PPO가 그 reward를 과도하게 최적화하면 실제 품질과 괴리가 생길 수 있다.

즉 PPO는 가장 일반적이지만, 그만큼 **“돌아가게 만드는 일” 자체가 기술**이다.

### 11.3 GRPO

GRPO는 PPO보다 구성은 단순해지지만, 다른 종류의 민감도가 생긴다.

- group size `G`
- 그룹 내부 reward 분산
- KL coefficient
- clip range
- completion 길이와 길이 편향
- 보상 정규화 방식

특히 group reward의 분산이 거의 없거나, 모든 샘플이 다 비슷하게 나쁘면 학습 신호가 약해질 수 있다. 또한 group sample 수를 늘리면 신호는 좋아질 수 있지만 토큰 비용이 빠르게 올라간다.

### 실전 감각 요약

- **안정성과 단순성**: DPO 우위
- **일반성과 유연성**: PPO 우위
- **reasoning용 온라인 RL의 효율**: GRPO 우위

단, 이것은 절대적인 우열이 아니라 **문제 설정이 다르기 때문에 생기는 상대적 장단점**이다.

---

## 12. reasoning / math / code 과제에서 왜 GRPO가 특히 강하게 거론되는가

이 질문은 매우 자주 나온다.

핵심 이유는 reasoning 과제에서는 **같은 문제에 대해 여러 풀이를 뽑아 비교하는 것 자체가 자연스럽기 때문**이다.

예를 들어 수학 문제 하나에 대해,

- 풀이 A는 길지만 맞다.
- 풀이 B는 짧지만 틀렸다.
- 풀이 C는 중간에 논리가 무너진다.
- 풀이 D는 정답은 맞지만 형식이 틀렸다.

이런 상황에서는 **same-prompt multi-sampling**이 매우 유의미하다. GRPO는 바로 그 구조를 advantage 계산에 녹인다.

또한 reasoning 과제는 다음 특성이 있다.

- 최종 정답 검증이 가능할 수 있다.
- 중간 단계(process step)에 대한 보상도 설계할 수 있다.
- 한 문제에 대해 여러 후보를 만들어 비교하는 것이 자연스럽다.

이 때문에 GRPO는 reasoning RL에서 특히 설득력이 크다.

반대로 DPO는 reasoning 과제에서도 쓸 수는 있지만, 보통은

- 좋은/나쁜 풀이를 **쌍 데이터로 미리 만들어야** 하고,
- 학습 중에 새로운 풀이를 탐색하면서 개선하는 구조는 아니다.

즉 reasoning처럼 **탐색과 검증**이 중요한 과제에서는 PPO/GRPO가 더 자연스럽고, 그중에서도 critic 부담을 줄이려는 방향에서 GRPO가 매력적이다.

---

## 13. DPO가 PPO를 완전히 대체하는가? 그렇지는 않다

이 부분은 균형 있게 봐야 한다.

DPO는 분명 많은 장점이 있다.

- 구현이 단순하고,
- reward model과 value model이 필요 없으며,
- preference dataset만 있으면 강력한 baseline이 된다.

하지만 다음 상황에서는 PPO/GRPO 계열이 여전히 필요하거나 더 자연스럽다.

1. **학습 중 온라인 탐색이 필요할 때**  
   고정된 데이터셋을 넘어서 현재 정책이 직접 새 답을 만들어 보고, 그 결과를 바탕으로 개선해야 할 때.

2. **programmatic reward / verifier reward를 직접 쓰고 싶을 때**  
   unit test 통과, 정답 여부, 포맷 검사, 실행 성공 여부 등.

3. **process reward가 중요할 때**  
   최종 결과뿐 아니라 중간 reasoning step 자체를 보상하고 싶을 때.

즉 DPO는 PPO 기반 RLHF의 강력한 대안이지만, **온라인 RL이 필요한 설정 전체를 대체하는 것은 아니다.**

---

## 14. GRPO가 PPO를 완전히 대체하는가? 이것도 아니다

GRPO는 PPO보다 메모리 효율적일 수 있고, reasoning 과제에 매우 잘 맞지만, PPO 전체를 완전히 대체한다고 보는 것도 과장이다.

### PPO가 여전히 가지는 장점

- 일반적인 RL 프레임워크로서 더 넓은 범용성
- critic/value estimation을 통한 전통적 advantage 추정
- 이미 축적된 구현/이론/실험 관행

### GRPO가 특히 유리한 지점

- value model 학습 비용을 줄이고 싶을 때
- same-prompt multi-sample 비교가 자연스러운 과제일 때
- reasoning / verifier 중심 reward가 있을 때

즉 PPO와 GRPO는 “구세대/신세대” 관계라기보다,

- **PPO는 일반적이고 무거운 표준형**
- **GRPO는 특정한 LLM reasoning 설정에 맞춘 효율형**

으로 보는 편이 정확하다.

---

## 15. 원 논문과 오피셜 구현(특히 TRL) 사이에서 꼭 알아야 할 차이

이 섹션은 실제 구현할 때 매우 중요하다.

### 15.1 PPO의 오피셜 구현 관점

Hugging Face TRL의 PPO 문서는 LLM RLHF용 PPO trainer가 보통 다음 구성요소를 요구한다고 명확히 보여 준다.

- policy model
- reference model
- reward model
- value model

즉 실무에서 “PPO로 RLHF 하자”는 말은 대개 **그냥 PPO 논문 하나만 구현하는 것**이 아니라, **LLM용 RLHF 파이프라인 전체를 구성**한다는 뜻에 가깝다.

### 15.2 DPO의 오피셜 구현 관점

TRL DPO 문서는 DPO를

- 비교적 안정적이고,
- 계산적으로 가볍고,
- `prompt`, `chosen`, `rejected` 형식의 preference dataset으로 바로 학습 가능한 방법

으로 정리한다.

즉 오피셜 구현 관점에서도 DPO의 정체성은 분명하다.

> **reward model 없는 직접 선호최적화**

### 15.3 GRPO의 오피셜 구현 관점: 논문과 1:1 동일하다고 보면 안 된다

이 부분이 특히 중요하다.

TRL의 GRPO 문서는 다음을 설명한다.

- GRPO는 온라인 학습 알고리즘이다.
- 프롬프트마다 여러 completion을 샘플링한다.
- reward model 또는 reward function으로 점수를 계산한다.
- group-relative advantage를 계산한다.
- KL approximator를 사용한다.

그런데 동시에, 현재 구현은 원래 DeepSeekMath 논문과 **완전히 동일한 세부식은 아닐 수 있다.** 대표적으로 다음이 알려져 있다.

- 원 논문과 달리 **length bias 문제 때문에** `1/|o_i|` 스케일링을 두지 않는 구현 차이가 있다.
- `beta=0.0`이 기본값인 구현이 있어, **기본 설정으로는 KL term이 사실상 꺼져 있을 수 있다.**

즉 아주 중요한 실전 교훈은 다음이다.

> **“GRPO를 쓴다”는 말만으로는 부족하다. 논문의 GRPO인지, 현재 라이브러리 기본값의 GRPO인지, length normalization과 KL beta가 어떻게 되어 있는지까지 확인해야 한다.**

이 차이는 재현성, 성능 비교, ablation 해석에 직접 영향을 준다.

---

## 16. 흔한 오해 정리

### 오해 1. DPO는 그냥 chosen 응답만 SFT 하는 것이다

아니다.

DPO는 단순히 chosen의 log-prob를 올리는 게 아니라,

- chosen과 rejected를 함께 보고,
- reference 대비 상대 마진을 학습한다.

즉 **pairwise comparative objective**가 핵심이다.

### 오해 2. GRPO는 DPO에 group 개념만 추가한 것이다

아니다.

GRPO는 본질적으로 **온라인 RL**이다.  
DPO는 **오프라인 preference optimization**이다.

둘 다 비교를 쓰지만, 비교의 위치와 학습 루프가 다르다.

### 오해 3. value model이 없으니 GRPO는 항상 PPO보다 낫다

그렇지 않다.

- PPO는 더 일반적이고,
- critic 기반 advantage가 더 잘 맞는 설정도 있으며,
- GRPO는 group sampling 비용과 reward normalization 민감도가 있다.

### 오해 4. DPO가 가장 단순하므로 모든 경우에 최선이다

아니다.

온라인 탐색, verifier reward, process supervision이 중요하면 PPO/GRPO가 더 자연스럽다.

### 오해 5. PPO의 핵심은 reference model이다

일반 PPO의 핵심은 아니다.  
PPO의 핵심은 **old policy 대비 clipped update**다.

reference model은 주로 **LLM RLHF라는 응용 맥락**에서 KL regularization용으로 들어온다.

---

## 17. 어떤 상황에서 무엇을 선택할까

### DPO를 우선 고려할 상황

- 이미 괜찮은 quality의 preference pair가 있다.
- RLHF 파이프라인 전체를 구축하기엔 비용이 크다.
- 빠르고 안정적인 alignment baseline이 필요하다.
- reward model / critic / rollout infrastructure를 줄이고 싶다.

### PPO를 우선 고려할 상황

- 온라인 RL이 필요하다.
- reward model 또는 환경 상호작용을 직접 최적화해야 한다.
- sequence-level / token-level reward를 자유롭게 설계하고 싶다.
- 범용적인 RL 프레임워크가 필요하다.

### GRPO를 우선 고려할 상황

- reasoning, math, code처럼 same-prompt multi-sample이 자연스럽다.
- verifier reward나 correctness reward가 있다.
- PPO의 critic 비용을 줄이고 싶다.
- 온라인 RL을 유지하면서도 value model 없이 학습하고 싶다.

### 가장 실용적인 의사결정 규칙

1. **고정된 preference pair만 있고 단순·안정성이 최우선** → **DPO**  
2. **온라인 reward 최적화와 최대 범용성이 필요** → **PPO**  
3. **reasoning 중심 온라인 RL이고 critic 비용을 줄이고 싶음** → **GRPO**

---

## 18. 최종 결론

세 방법은 서로의 단순한 상위호환/하위호환 관계가 아니다.

- **DPO**는 RLHF의 목적을 **오프라인 pairwise preference optimization**으로 직접 푼다.
- **PPO**는 reward를 최대화하는 **표준 온라인 RL 프레임워크**다.
- **GRPO**는 PPO 계열 안에서 **critic을 제거하고 group-relative comparison을 활용**하는 LLM 친화적 변형이다.

가장 중요한 차이는 다음 네 줄로 요약할 수 있다.

1. **DPO는 pairwise preference 데이터 중심**이다.  
2. **PPO는 reward/value/rollout 중심**이다.  
3. **GRPO는 grouped rollout / relative reward 중심**이다.  
4. **문제에 따라 정답이 달라지지, 셋 중 하나가 항상 우월한 것은 아니다.**

실무적으로는 보통 다음처럼 이해하면 된다.

- **정적 선호 정렬**: DPO가 강함  
- **가장 일반적인 온라인 RL**: PPO가 강함  
- **reasoning용 online RL 효율화**: GRPO가 강함

---

## 19. 참고문헌 및 오피셜 자료

### 원 논문

1. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**  
   Rafael Rafailov, Archit Sharma, Eric Mitchell, et al.  
   URL: <https://arxiv.org/abs/2305.18290>

2. **Proximal Policy Optimization Algorithms**  
   John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov.  
   URL: <https://arxiv.org/abs/1707.06347>

3. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**  
   Zihong Shao, Peiyi Wang, Qihua Zhu, et al.  
   URL: <https://arxiv.org/abs/2402.03300>

4. **Training language models to follow instructions with human feedback (InstructGPT)**  
   Long Ouyang, Jeffrey Wu, Xu Jiang, et al.  
   URL: <https://arxiv.org/abs/2203.02155>

### 오피셜 / 준오피셜 문서

5. **OpenAI Spinning Up — PPO**  
   URL: <https://spinningup.openai.com/en/latest/algorithms/ppo.html>

6. **Hugging Face TRL — DPO Trainer**  
   URL: <https://huggingface.co/docs/trl/dpo_trainer>

7. **Hugging Face TRL — PPO Trainer**  
   URL: <https://huggingface.co/docs/trl/ppo_trainer>

8. **Hugging Face TRL — GRPO Trainer**  
   URL: <https://huggingface.co/docs/trl/grpo_trainer>

---
