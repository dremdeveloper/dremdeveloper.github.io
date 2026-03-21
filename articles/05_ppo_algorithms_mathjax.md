---
title: "Proximal Policy Optimization Algorithms"
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

## 핵심 요약

- PPO는 정책을 한 번에 너무 크게 바꾸지 않도록 막아 **정책 경사 기반 강화학습을 안정화**한 알고리즘이다.
- 핵심 아이디어는 오래된 정책과 새 정책의 비율을 이용한 **clipped surrogate objective**에 있다.
- TRPO보다 계산이 간단하면서도 업데이트 폭을 제어할 수 있어, 이후 RLHF 계열에서도 널리 쓰이게 되었다.
- 실제로는 보상 스케일, KL 제어, advantage 추정 방식에 따라 결과가 크게 달라지므로 구현 세부가 중요하다.

# Proximal Policy Optimization Algorithms

## 확장 해설
## 문헌 정보

- **논문명**: *Proximal Policy Optimization Algorithms*
- **저자**: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
- **소속**: OpenAI
- **버전**: arXiv:1707.06347v2, 2017-08-28

본 문서는 원 논문의 구조를 유지하면서, 각 절과 수식, 표, 그림 캡션, 각주, 부록, 참고문헌까지 빠짐없이 서술하는 해설본이다. 기술적 설명은 원문 이해를 돕기 위해 보강하되, 논문에 명시되지 않은 사항은 논문 주장인 것처럼 단정하지 않는다. 절, 식, 그림, 표의 번호는 모두 원 논문 기준을 따른다.

---

## Abstract

논문 초록은 PPO의 문제의식, 방법론, 실험 범위, 그리고 저자들이 주장하는 실질적 장점을 매우 압축적으로 제시한다.

저자들은 강화학습에서 **환경과의 상호작용을 통해 데이터를 수집하고**, 그 데이터 위에서 **대리 목적함수(surrogate objective)**를 **확률적 경사상승(stochastic gradient ascent)**으로 최적화하는 새로운 정책경사 계열의 방법군을 제안한다. 여기서 핵심은 샘플링과 최적화가 교대로 이루어진다는 점이다. 즉, 정책으로부터 데이터를 얻은 뒤 그 데이터에 대해 정책을 개선하고, 다시 새 정책으로 데이터를 수집하는 절차가 반복된다.

기존의 표준 정책경사 방법은 일반적으로 **데이터 샘플 하나당 한 번의 gradient update**만 수행한다. 이에 비해 저자들은 **하나의 데이터 배치에 대해 minibatch 기준으로 여러 epoch의 업데이트를 수행할 수 있게 해주는 새로운 목적함수**를 도입한다. 이 방법군을 저자들은 **Proximal Policy Optimization(PPO)**이라 명명한다.

초록은 PPO가 **Trust Region Policy Optimization(TRPO)**의 장점을 일부 계승한다고 설명한다. 다만 이 논문은 단순히 “TRPO와 비슷하다”는 수준에 머물지 않고, PPO가 **구현이 훨씬 간단하고**, **더 일반적인 설정에 적용 가능하며**, **경험적으로 더 나은 sample complexity를 보인다**고 주장한다.

실험은 시뮬레이션 로봇 보행과 Atari 게임을 포함한 다수의 벤치마크에서 수행된다. 초록에 따르면 PPO는 다른 online policy gradient 방법보다 우수한 성능을 보이며, 궁극적으로 **sample complexity, 단순성, wall-time 사이의 균형**이 좋다는 점을 보여준다.

---

## 1. Introduction

서론은 신경망 함수근사를 사용하는 강화학습의 대표적 접근들을 정리한 뒤, PPO가 해결하고자 하는 공백이 무엇인지 명확히 제시한다.

최근 수년간 신경망 기반 강화학습에서는 여러 접근이 제안되었다. 저자들은 그중 대표적 경쟁 후보로 **deep Q-learning**, **vanilla policy gradient**, **trust region / natural policy gradient** 계열을 지목한다. 그러나 이들 모두가 실용적 관점에서 충분하다고 보지는 않는다. 저자들이 이상적인 방법에 요구하는 성질은 세 가지다. 첫째, **대규모 모델과 병렬 구현에 대해 확장 가능해야 한다**. 둘째, **데이터 효율적이어야 한다**. 셋째, **하이퍼파라미터 튜닝 없이도 다양한 문제에서 안정적으로 성공할 정도로 강건해야 한다**.

이 기준 아래에서 저자들은 각 방법의 한계를 비교한다.

- **Q-learning with function approximation**은 많은 단순한 문제에서도 실패하며, 이론적으로도 충분히 이해되지 않았다고 지적된다.
- **Vanilla policy gradient**는 데이터 효율과 강건성이 부족하다고 진단된다.
- **TRPO**는 상대적으로 복잡하고, dropout과 같은 노이즈를 포함하는 아키텍처나 정책과 가치함수 사이의 파라미터 공유, 혹은 보조 과제와의 파라미터 공유 같은 일반적 딥러닝 설계와 잘 맞지 않는다고 평가된다.

서론의 각주 1은 이 비판을 구체화한다. DQN은 ALE와 같은 **이산 행동공간 게임 환경**에서는 우수한 성능을 보이지만, OpenAI Gym이나 Duan et al.이 정리한 연속제어 벤치마크에서는 좋은 성능이 입증되지 않았다는 것이다. 즉, 저자들이 보기에 DQN류 방법은 연속제어까지 포괄하는 범용 강화학습 해법으로 제시되기 어렵다.

이러한 문제의식 위에서, 논문은 **TRPO의 데이터 효율과 신뢰할 수 있는 성능을 유지하면서도 1차 최적화만 사용하는 알고리즘**을 제안하는 것을 목표로 한다. 그 중심에는 **확률비율(probability ratio)을 clipping한 새로운 목적함수**가 놓인다. 저자들은 이 목적함수가 정책 성능에 대한 **비관적 추정**, 즉 **하한(lower bound)**의 역할을 한다고 설명한다. 최적화 절차는 정책으로부터 데이터를 수집한 뒤, 그 데이터를 사용하여 여러 epoch 동안 정책을 최적화하는 방식으로 설계된다.

서론 말미에서 저자들은 실험의 방향도 예고한다. 먼저 서로 다른 surrogate objective 변형들을 비교하여 **clipped probability ratio를 사용하는 버전이 가장 우수함**을 보였다고 밝힌다. 이어 기존 알고리즘들과의 비교에서는, 연속제어에서는 PPO가 비교 대상보다 우수하며, Atari에서는 A2C보다 sample complexity 면에서 현저히 우수하고 ACER과 유사한 수준이지만 훨씬 단순하다고 주장한다.

---

## 2. Background: Policy Optimization

2절은 PPO의 동기를 수식적으로 준비하는 절이다. 2.1절은 정책경사(policy gradient)의 기본 형태를 정리하고, 2.2절은 TRPO가 어떻게 정책 업데이트의 크기를 제약하는지 설명한다. PPO는 바로 이 두 흐름의 접점에서 출발한다.

### 표기 및 기호

논문 전반에서 자주 등장하는 표기는 다음과 같다.

| 기호 | 의미 |
|---|---|
| \(s_t\) | 시각 \(t\)의 상태 |
| \(a_t\) | 시각 \(t\)의 행동 |
| \(\pi_\theta(a_t \mid s_t)\) | 파라미터 \(\theta\)를 갖는 확률적 정책 |
| \(\theta_{\text{old}}\) | 업데이트 이전 정책의 파라미터 |
| \(\hat{\mathbb{E}}_t[\cdot]\) | 유한한 샘플 배치에 대한 경험적 평균 |
| \(\hat A_t\) | 시각 \(t\)에서의 advantage 추정치 |
| \(r_t(\theta)\) | 확률비율 \(\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}\) |
| \(r_t\) | 환경으로부터 얻는 즉시 보상(reward) |
| \(KL[\pi_{\theta_{\text{old}}}, \pi_\theta]\) | 이전 정책과 현재 정책 사이의 KL divergence |
| \(\beta\) | KL penalty 계수 |
| \(\epsilon\) | clipping 범위를 정하는 하이퍼파라미터 |
| \(V(s)\) | 상태가치함수 |
| \(\gamma\) | 할인율 |
| \(\lambda\) | GAE 파라미터 |

특히 \(r_t\)는 문맥에 따라 **보상(reward)**과 **확률비율(probability ratio)**이라는 서로 다른 의미로 사용되므로 해석 시 주의가 필요하다. 3절에서 등장하는 \(r_t(\theta)\)는 ratio이며, 5절의 식 (10)–(12)에 등장하는 \(r_t\)는 reward이다.

### 2.1. Policy Gradient Methods

정책경사 방법은 정책의 기울기(policy gradient)를 추정하고, 이 추정치를 확률적 경사상승 알고리즘에 입력하여 정책을 개선하는 방식으로 작동한다. 논문이 제시하는 가장 일반적인 gradient estimator는 다음과 같다.

$$
\hat g
=
\hat{\mathbb E}_t
\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\hat A_t
\right].
\tag{1}
$$

여기서 \(\pi_\theta\)는 확률적 정책이며, \(\hat A_t\)는 시각 \(t\)에서의 advantage function 추정치이다. \(\hat{\mathbb E}_t[\cdot]\)는 무한한 기대값이 아니라, 샘플링과 최적화를 교대로 수행하는 알고리즘 안에서 얻어진 **유한 배치에 대한 경험적 평균**을 뜻한다.

자동미분 소프트웨어를 사용하는 구현에서는, 직접 식 (1)을 계산하기보다 그 gradient가 곧 policy gradient estimator가 되도록 목적함수를 구성한다. 논문은 다음 목적함수를 제시한다.

$$
L^{PG}(\theta)
=
\hat{\mathbb E}_t
\left[
\log \pi_\theta(a_t \mid s_t)\,\hat A_t
\right].
\tag{2}
$$

이때 \(\hat g\)는 \(L^{PG}\)를 \(\theta\)에 대해 미분함으로써 얻어진다. 즉, 정책경사를 “목적함수의 미분” 형태로 구현하는 것이 일반적인 딥러닝 프레임워크와 잘 결합된다는 점을 강조하는 대목이다.

문제는 여기서 시작된다. 식 (2)의 loss에 대해 **같은 trajectory를 여러 번 사용하여 반복적으로 최적화**하는 것은 직관적으로 매력적이지만, 저자들은 이것이 충분히 정당화되지 않으며 경험적으로도 **파괴적으로 큰 정책 업데이트**를 자주 일으킨다고 말한다. 논문은 Section 6.1을 참조시키며, 결과는 본문에 직접 도표로 제시하지 않았지만 “no clipping or penalty” 조건과 유사하거나 더 나빴다고 설명한다. 이 진술은 PPO의 중심 문제가 무엇인지 분명히 보여준다. 즉, 하나의 on-policy 데이터 배치를 여러 번 재사용하려면, 정책이 한 번에 너무 멀리 변하지 않도록 제어할 장치가 필요하다.

### 2.2. Trust Region Methods

TRPO는 바로 이 문제를 해결하려는 대표적 접근으로 소개된다. TRPO에서는 “surrogate objective”를 최대화하되, 정책 업데이트의 크기에 대한 제약을 함께 둔다.

TRPO의 목적은 다음과 같다.

$$
\max_\theta
\hat{\mathbb E}_t
\left[
\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
\hat A_t
\right]
\tag{3}
$$

subject to

$$
\hat{\mathbb E}_t
\left[
KL\big[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)\big]
\right]
\le \delta.
\tag{4}
$$

여기서 \(\theta_{\text{old}}\)는 업데이트 이전 정책 파라미터를 의미한다. 목적함수는 기존 정책으로부터 얻은 데이터를 현재 정책 아래에서 재가중하는 형태이며, 제약식은 정책이 평균적으로 어느 정도까지 바뀔 수 있는지를 KL divergence로 제한한다.

논문에 따르면 이 문제는 목적함수를 선형 근사하고 제약을 이차 근사한 뒤 **conjugate gradient algorithm**을 사용하여 효율적으로 근사적으로 풀 수 있다. 즉, TRPO는 단순한 SGD가 아니라 제약 조건을 동반한 보다 정교한 최적화 절차를 요구한다.

흥미로운 점은, TRPO를 정당화하는 이론이 실제로는 hard constraint보다 **penalty formulation**을 더 직접적으로 시사한다는 것이다. 논문은 다음의 unconstrained optimization problem을 제시한다.

$$
\max_\theta
\hat{\mathbb E}_t
\left[
\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}\hat A_t
-
\beta\,
KL\big[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)\big]
\right].
\tag{5}
$$

여기서 \(\beta\)는 penalty 계수이다. 이 정식화는 어떤 surrogate objective가 정책 성능에 대한 lower bound, 즉 pessimistic bound를 형성한다는 사실에 근거하며, 특히 그 이론적 surrogate는 평균 KL이 아니라 **상태별 최대 KL(max KL over states)**를 사용한다고 논문은 설명한다.

그럼에도 불구하고 TRPO가 hard constraint를 유지하는 이유는 매우 실용적이다. 고정된 \(\beta\) 하나를 선택하여 모든 문제에서 잘 작동하게 만드는 것이 어렵고, 심지어 하나의 문제 안에서도 학습 과정에 따라 적절한 \(\beta\) 값이 변하기 때문이다. 따라서 저자들의 목표인 “TRPO의 단조 개선(monotonic improvement)을 1차 최적화로 모사하는 알고리즘”을 얻기 위해서는, 단순히 식 (5)에 고정 penalty를 두고 SGD를 수행하는 것만으로는 충분하지 않다. 논문은 명시적으로 **추가적인 수정이 필요하다**고 결론짓는다. PPO는 바로 그 수정의 한 형태로 제시된다.

---

## 3. Clipped Surrogate Objective

3절은 PPO의 핵심을 제시하는 절이다. TRPO가 제약을 통해 정책 변화량을 제한했다면, PPO는 목적함수 자체를 수정하여 **너무 큰 정책 변화가 더 이상 이득을 주지 않도록 만드는 방식**을 택한다.

먼저 확률비율을 다음과 같이 정의한다.

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}.
$$

따라서 \(\theta = \theta_{\text{old}}\)일 때는 \(r_t(\theta_{\text{old}})=1\)이다.

TRPO가 최대화하는 surrogate objective는 다음과 같이 다시 쓸 수 있다.

$$
L^{CPI}(\theta)
=
\hat{\mathbb E}_t
\left[
\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}\hat A_t
\right]
=
\hat{\mathbb E}_t
\left[
r_t(\theta)\hat A_t
\right].
\tag{6}
$$

여기서 상첨자 \(CPI\)는 conservative policy iteration을 가리킨다. 이 목적함수만을 제약 없이 최대화하면 정책이 과도하게 멀리 이동할 수 있으므로, 논문은 \(r_t(\theta)\)가 1에서 멀어지는 움직임을 제어하기 위해 목적함수를 다음과 같이 수정한다.

$$
L^{CLIP}(\theta)
=
\hat{\mathbb E}_t
\left[
\min\Big(
r_t(\theta)\hat A_t,\;
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\Big)
\right].
\tag{7}
$$

논문은 \(\epsilon\)의 예시값으로 \(0.2\)를 든다. 식 (7)의 구조를 해부하면 다음과 같다.

1. 첫 번째 항 \(r_t(\theta)\hat A_t\)는 기존의 \(L^{CPI}\)와 동일한 항이다.
2. 두 번째 항은 ratio \(r_t(\theta)\)를 \([1-\epsilon, 1+\epsilon]\) 구간 안으로 clipping한 것이다.
3. 마지막으로 둘 중 **최솟값(minimum)**을 취함으로써, 최종 목적함수가 unclipped objective의 **lower bound**, 즉 비관적 bound가 되도록 만든다.

이 구성의 의미는 논문이 직접 설명하듯 분명하다. 확률비율 변화가 목적함수를 **개선**시킬 때에는 그 이익을 일정 구간 밖에서는 더 이상 인정하지 않고, 반대로 변화가 목적함수를 **악화**시킬 때에는 그 악화를 그대로 반영한다. 결과적으로 큰 정책 이동이 가져올 수 있는 과도한 낙관을 억제하게 된다.

또한 논문은 \(L^{CLIP}(\theta)\)가 \(\theta_{\text{old}}\) 주변에서는 \(L^{CPI}(\theta)\)와 **1차 근사 수준에서 동일**하다고 설명한다. 즉, \(r=1\) 근방에서는 둘의 gradient 방향이 사실상 같지만, \(\theta\)가 \(\theta_{\text{old}}\)에서 멀어질수록 두 목적함수는 달라진다. 이 점은 PPO가 기존 정책경사의 국소적인 업데이트 방향은 유지하면서도, 과도한 이동을 스스로 억제한다는 사실을 보여준다.

### Figure 1 해설

> Figure 1 삽입

Figure 1은 \(L^{CLIP}\)의 단일 시점 항을 \(r\)의 함수로 시각화한다. 왼쪽은 \(\hat A_t > 0\)인 경우, 오른쪽은 \(\hat A_t < 0\)인 경우이며, 두 그림 모두 빨간 원은 최적화 시작점 \(r=1\)을 표시한다.

- \(\hat A_t > 0\)인 경우, 해당 행동의 확률을 늘리는 것이 유리하므로 \(r\)이 증가할수록 목적함수도 증가한다. 그러나 \(r > 1+\epsilon\)에 이르면 clipping된 항이 작동하여 목적함수가 더 이상 증가하지 않고 평평해진다.
- \(\hat A_t < 0\)인 경우, 해당 행동의 확률을 줄이는 것이 유리하므로 \(r\)이 감소할수록 목적함수가 개선된다. 그러나 \(r < 1-\epsilon\)로 더 내려가더라도 목적함수는 더 이상 개선되지 않는다.

이 그림은 clipping이 “정책 업데이트를 금지”하는 것이 아니라, **일정 범위를 넘어서는 업데이트에 대한 추가 인센티브를 제거**한다는 점을 시각적으로 보여준다. 논문은 또한 \(L^{CLIP}\)이 이러한 단일 항들을 많이 합한 형태라는 점도 함께 강조한다.

### Figure 2 해설

> Figure 2 삽입

Figure 2는 \(\theta_{\text{old}}\)와 한 번의 PPO 업데이트 이후 파라미터 사이를 선형 보간하면서, 몇 가지 surrogate objective가 어떻게 변화하는지를 보여준다. 이 그림은 연속제어의 Hopper-v1 문제에서 **첫 번째 policy update**에 해당하며, 업데이트된 정책은 초기 정책에 대해 약 **0.02의 KL divergence**를 갖는다. 논문은 바로 이 지점에서 \(L^{CLIP}\)이 최대가 된다고 설명한다.

그림이 전달하는 바는 명확하다. \(L^{CPI}\)는 업데이트 방향으로 계속 증가할 수 있지만, \(L^{CLIP}\)은 일정 시점 이후 증가를 멈추고 오히려 감소하여 “지나치게 큰 정책 업데이트”를 억제하는 효과를 낸다. 다시 말해, Figure 2는 \(L^{CLIP}\)이 \(L^{CPI}\)의 lower bound처럼 작동하며, 정책 이동이 과도할 경우 penalty와 유사한 역할을 수행한다는 직관을 제공한다.

## 4. Adaptive KL Penalty Coefficient

4절은 clipping과는 별개의 대안으로서, 혹은 clipping에 추가하는 방식으로서 **KL divergence penalty**를 사용하는 PPO 변형을 다룬다. 중요한 점은 저자들이 이 절을 제시하면서도, 실험적으로는 **KL penalty가 clipped surrogate objective보다 성능이 나빴다**고 명시한다는 사실이다. 그럼에도 불구하고 이를 논문에 포함한 이유는, 이 변형이 중요한 baseline이기 때문이다.

저자들이 제시하는 가장 단순한 형태의 알고리즘은 각 policy update마다 다음 절차를 수행한다.

첫째, 여러 epoch의 minibatch SGD를 사용하여 다음의 KL-penalized objective를 최적화한다.

$$
L^{KLPEN}(\theta)
=
\hat{\mathbb E}_t
\left[
\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}\hat A_t
-
\beta\,
KL\big[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)\big]
\right].
\tag{8}
$$

둘째, 실제 업데이트 후 평균 KL divergence를

$$
d
=
\hat{\mathbb E}_t
\left[
KL\big[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)\big]
\right]
$$

로 계산한다.

셋째, 목표 KL 값 \(d_{\text{targ}}\)를 유지하도록 \(\beta\)를 조정한다.

- 만약 \(d < d_{\text{targ}}/1.5\)이면 \(\beta \leftarrow \beta/2\)
- 만약 \(d > d_{\text{targ}}\times 1.5\)이면 \(\beta \leftarrow \beta \times 2\)

마지막으로 조정된 \(\beta\)를 다음 policy update에 사용한다.

논문은 이 규칙이 완벽히 정확한 제어를 제공하는 것은 아니라고 인정한다. 때로는 KL divergence가 목표값과 상당히 다르게 나타나기도 한다. 그러나 그러한 경우는 드물고, \(\beta\)가 빠르게 조정되기 때문에 전체 알고리즘은 안정적으로 작동한다고 설명한다. 또한 1.5와 2라는 수치는 휴리스틱하게 선택되었지만 성능이 이에 매우 민감하지 않으며, \(\beta\)의 초기값도 알고리즘이 빠르게 적응하므로 실무적으로 중요하지 않다고 말한다.

이 절의 의의는 두 가지다. 하나는 “고정 penalty coefficient는 어렵다”는 2절의 문제의식을 실제 적응 규칙으로 구현했다는 점이다. 다른 하나는, 그럼에도 불구하고 최종적으로는 clipping 방식이 더 나은 실험 결과를 보였다는 사실을 통해 PPO의 표준형이 왜 clip 버전으로 정착했는지를 설명해 준다는 점이다.

---

## 5. Algorithm

5절은 앞서 소개한 surrogate loss들이 실제 정책경사 코드 안에서 어떻게 사용되는지를 구체적인 학습 절차로 연결한다.

논문은 먼저 3절과 4절의 surrogate loss가 **전형적인 policy gradient 구현에 매우 작은 수정만으로도 계산 및 미분 가능**하다고 설명한다. 자동미분 기반 구현에서는 \(L^{PG}\) 대신 \(L^{CLIP}\) 또는 \(L^{KLPEN}\)을 손실로 구성하고, 그 목적함수에 대해 여러 단계의 stochastic gradient ascent를 수행하면 된다.

### 가치함수와 엔트로피 항을 포함한 결합 목적함수

분산 감소형 advantage estimator들은 일반적으로 학습된 상태가치함수 \(V(s)\)를 사용한다. 논문은 그 예로 generalized advantage estimation(GAE)과 Mnih et al.의 finite-horizon estimator를 든다. 만약 정책과 가치함수가 하나의 신경망에서 파라미터를 공유한다면, 정책 surrogate만으로는 충분하지 않으며 **가치함수 오차 항(value function error term)**을 함께 포함한 목적함수가 필요하다. 또한 과거 연구를 따라 충분한 exploration을 보장하기 위해 **entropy bonus**를 추가할 수 있다.

이들을 결합한 목적함수는 다음과 같다.

$$
L_t^{CLIP+VF+S}(\theta)
=
\hat{\mathbb E}_t
\left[
L_t^{CLIP}(\theta)
-
c_1 L_t^{VF}(\theta)
+
c_2 S[\pi_\theta](s_t)
\right],
\tag{9}
$$

여기서 \(c_1, c_2\)는 계수이며, \(S\)는 entropy bonus를 뜻한다. \(L_t^{VF}\)는 다음과 같은 제곱오차이다.

$$
L_t^{VF}(\theta)
=
\big(V_\theta(s_t)-V_t^{\text{targ}}\big)^2.
$$

식 (9)은 각 iteration마다 **대략적으로 최대화**되는 목적함수라고 서술된다. 즉, PPO는 단순한 policy loss만이 아니라 정책 개선, 가치 추정, 탐색 유지라는 세 요소를 동시에 관리하는 actor-critic 구조 안에서 이해되어야 한다.

### 고정 길이 trajectory segment와 advantage estimator

논문은 Mnih et al.에 의해 널리 사용된 구현 방식을 소개한다. 이 방식은 정책을 episode 전체 길이만큼 실행하지 않고, **episode 길이보다 훨씬 작은 \(T\) timestep 동안만 실행한 뒤**, 수집된 샘플을 사용하여 업데이트한다. 이 설계는 recurrent neural network와 결합하기에 특히 적합하다고 논문은 설명한다. 그러나 이 경우 advantage estimator가 \(T\) 시점을 넘어가는 정보를 사용하지 않도록 만들어야 한다.

Mnih et al.이 사용한 estimator는 다음과 같다.

$$
\hat A_t
=
- V(s_t)
+ r_t + \gamma r_{t+1} + \cdots + \gamma^{T-t-1}r_{T-1}
+ \gamma^{T-t}V(s_T).
\tag{10}
$$

여기서 \(t\)는 길이 \(T\)의 trajectory segment 내부에서의 time index이며, \(t \in [0,T]\)로 이해된다. 이 식은 현재 상태가치 \(V(s_t)\)를 baseline으로 빼고, 길이 \(T\) 구간 안에서의 할인 누적보상에 마지막 상태 \(s_T\)의 bootstrap value를 더한 형태다.

이 선택을 일반화한 것이 truncated GAE이다.

$$
\hat A_t
=
\delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t-1}\delta_{T-1},
\tag{11}
$$

여기서

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
\tag{12}
$$

논문은 식 (11)이 \(\lambda=1\)일 때 식 (10)으로 환원된다고 설명한다. 즉, PPO는 advantage estimator 자체를 새롭게 제안하는 논문이 아니라, 기존의 효과적인 advantage 추정기와 결합될 수 있는 정책최적화 목적함수 및 학습 절차를 제시하는 논문이다.

### Algorithm 1: PPO, Actor-Critic Style

고정 길이 trajectory segment를 사용하는 PPO 알고리즘은 다음과 같이 정리된다.

```text
Algorithm 1 PPO, Actor-Critic Style

for iteration = 1, 2, ... do
    for actor = 1, 2, ..., N do
        Run policy π_{θ_old} in environment for T timesteps
        Compute advantage estimates Â_1, ..., Â_T
    end for
    Optimize surrogate L w.r.t. θ, with K epochs and minibatch size M ≤ NT
    θ_old ← θ
end for
```

이 알고리즘의 구조는 단순하지만 중요하다.

1. \(N\)개의 병렬 actor가 각각 \(T\) timestep의 데이터를 수집한다.
2. 따라서 iteration마다 총 \(NT\) timestep의 on-policy 데이터가 모인다.
3. 이 데이터에 대해 surrogate loss를 구성한다.
4. minibatch SGD, 혹은 논문이 덧붙이듯 보통 더 나은 성능을 보이는 Adam을 사용하여 \(K\) epoch 동안 최적화한다.
5. 업데이트가 끝나면 현재 정책 파라미터를 \(\theta_{\text{old}}\)로 복사하여 다음 iteration의 기준 정책으로 삼는다.

이 절은 PPO가 왜 “구현이 간단하다”고 평가되는지를 잘 보여준다. 복잡한 제약최적화 절차 없이, 표준 actor-critic 학습 루프 안에 surrogate loss만 삽입하면 되기 때문이다.

---

## 6. Experiments

6절은 PPO의 empirical validation을 담당한다. 논문은 먼저 surrogate objective의 여러 변형을 비교하고, 이후 연속제어와 Atari 도메인에서 기존 방법들과 성능을 비교한다. 또한 고차원 연속제어 문제인 3D humanoid 실험을 통해 PPO의 확장성을 시연한다.

### 6.1. Comparison of Surrogate Objectives

이 실험의 목적은 PPO의 핵심 surrogate objective 가운데 어떤 형태가 가장 적절한지를 검증하는 데 있다. 논문은 \(L^{CLIP}\)을 몇 가지 자연스러운 변형 및 제거(ablation) 버전과 비교한다.

비교한 목적함수는 다음과 같다.

1. **No clipping or penalty**

   $$
   L_t(\theta) = r_t(\theta)\hat A_t
   $$

2. **Clipping**

   $$
   L_t(\theta)
   =
   \min\big(
   r_t(\theta)\hat A_t,\;
   \operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
   \big)
   $$

3. **KL penalty (fixed or adaptive)**

   $$
   L_t(\theta)
   =
   r_t(\theta)\hat A_t
   - \beta\, KL[\pi_{\theta_{\text{old}}},\pi_\theta]
   $$

각주 5는 KL penalty의 경우 \(\beta\)를 고정할 수도 있고, 4절에서 설명한 방식대로 target KL 값 \(d_{\text{targ}}\)에 따라 adaptive coefficient를 사용할 수도 있음을 명시한다. 또한 저자들은 **log space에서의 clipping도 시도했으나 성능이 더 좋지 않았다**고 덧붙인다.

하이퍼파라미터 탐색을 포함한 비교를 수행해야 하므로, 저자들은 계산 비용이 저렴한 벤치마크를 선택한다. 구체적으로 OpenAI Gym에 구현된 MuJoCo 기반의 **7개 simulated robotics task**를 사용하며, 각 실험은 **1 million timesteps** 동안 학습된다. 각주 2는 이 7개 환경이 **HalfCheetah, Hopper, InvertedDoublePendulum, InvertedPendulum, Reacher, Swimmer, Walker2d**이며 모두 “-v1” 버전임을 명시한다.

이 실험에서 탐색하는 하이퍼파라미터는 clipping의 \(\epsilon\), 그리고 KL penalty의 \(\beta\) 및 \(d_{\text{targ}}\)이며, 나머지 하이퍼파라미터는 Appendix A의 Table 3에 제시된다.

정책 표현은 두 개의 hidden layer를 가지는 fully connected MLP이며, 각 층은 64개의 unit과 tanh 비선형성을 사용한다. 출력은 가우시안 분포의 평균이고, 표준편차는 가변적(variable)이다. 이는 TRPO 및 연속제어 벤치마크 선행연구의 설정을 따른 것이다. 이 실험에서는 정책과 가치함수 사이에 파라미터를 공유하지 않으므로 \(c_1\)은 의미가 없고, entropy bonus도 사용하지 않는다.

평가 방식도 논문은 구체적으로 설명한다. 각 알고리즘은 7개 환경 모두에 대해 3개의 random seed로 실행된다. 각 run의 점수는 **마지막 100개 episode의 평균 total reward**로 정의한다. 이후 환경마다 점수를 shift 및 scale하여 random policy는 0점, 최상의 결과는 1점이 되도록 정규화하고, 총 21개 run의 평균을 내어 각 알고리즘 설정에 대한 단일 scalar score를 얻는다.

#### Table 1. Continuous Control Benchmark 결과

> Table 1 삽입

표 1은 clipped objective가 실험적으로 가장 우수하다는 논문의 핵심 주장에 직접적인 근거를 제공한다. 특히 \(\epsilon=0.2\)일 때 평균 정규화 점수가 0.82로 가장 높다. 반면 clipping이나 penalty를 전혀 사용하지 않은 설정은 -0.39라는 음수 점수를 기록하는데, 논문은 그 이유를 HalfCheetah 환경에서 매우 큰 음수 점수가 발생하여 초기 random policy보다도 나쁜 결과가 나왔기 때문이라고 설명한다. 이 결과는 같은 on-policy 데이터를 여러 번 재사용하는 상황에서 update constraint가 없을 경우 정책이 쉽게 붕괴할 수 있음을 강하게 시사한다.

또한 adaptive KL과 fixed KL 모두 일정 수준의 성능을 보이지만, clipping 계열의 최고 성능에는 미치지 못한다. 4절에서 저자들이 KL penalty를 “중요한 baseline”이라고는 했지만 주된 선택으로 제시하지 않은 이유가 여기서 실증적으로 확인된다.

표 1의 캡션은 추가로, 이 결과가 **7개 환경 × 3개 시드 = 21회 run**의 평균 정규화 점수이며, fixed KL 실험에서 \(\beta\)는 1로 초기화되었다고 밝힌다.

### 6.2. Comparison to Other Algorithms in the Continuous Domain

이 절에서는 3절의 clipped surrogate objective를 사용하는 PPO를 기존의 대표적 연속제어 알고리즘들과 비교한다. 비교 대상은 다음과 같다.

- TRPO
- Cross-Entropy Method(CEM)
- Vanilla policy gradient with adaptive stepsize
- A2C
- A2C with trust region

논문은 A2C가 advantage actor critic의 약자이며, A3C의 동기식(synchronous) 버전이라고 설명한다. 저자들은 비동기 버전보다 동기 버전이 같거나 더 나은 성능을 보인다고 판단하여 A2C를 사용했다. 각주 3은 adaptive stepsize를 사용하는 vanilla policy gradient에 대해, 각 batch 이후 원래 정책과 업데이트된 정책 사이의 KL divergence를 이용하여 Adam의 stepsize를 4절과 유사한 규칙으로 조정한다고 부연한다. 또한 해당 구현이 Berkeley Deep RL course의 공개 과제 저장소에 제공된다고 덧붙인다.

PPO는 직전 절의 하이퍼파라미터를 그대로 사용하되, clipping parameter는 \(\epsilon=0.2\)로 고정한다. 본문은 결과를 한 문장으로 요약한다. 즉, **PPO는 거의 모든 continuous control 환경에서 이전 방법들보다 우수하다**.

#### Figure 3 해설

> Figure 3 삽입

Figure 3은 HalfCheetah-v1, Hopper-v1, InvertedDoublePendulum-v1, InvertedPendulum-v1, Reacher-v1, Swimmer-v1, Walker2d-v1의 7개 MuJoCo 환경에 대해, 100만 timestep 동안의 학습 곡선을 제시한다. 캡션은 “several MuJoCo environments에서 여러 알고리즘을 비교하였고, 학습 길이는 one million timesteps”라고 명시한다.

시각적으로 보아도 PPO(Clip)는 여러 환경에서 가장 높은 곡선 혹은 최상위 곡선군에 위치한다. 특히 HalfCheetah-v1, Reacher-v1, Swimmer-v1, Walker2d-v1에서는 PPO가 비교적 명확한 우위를 보인다. InvertedDoublePendulum-v1과 InvertedPendulum-v1처럼 여러 방법이 빠르게 높은 점수에 도달하는 환경에서는 차이가 압도적이지 않지만, PPO 역시 상위권에 포함된다. 본문이 “almost all the continuous control environments”라고 서술한 이유를 Figure 3이 시각적으로 뒷받침한다.

이 절에서 중요한 점은 PPO가 단지 TRPO와 비슷한 안정성을 가진다는 수준을 넘어서, 튜닝된 경쟁 알고리즘들과의 직접 비교에서도 실질적인 성능 우위를 보였다는 것이다.

### 6.3. Showcase in the Continuous Domain: Humanoid Running and Steering

이 절은 PPO가 단순한 저차원 MuJoCo 벤치마크에만 국한되지 않고, 보다 어려운 **고차원 연속제어 문제**에도 적용 가능함을 보여주기 위한 “showcase” 성격의 실험이다.

저자들은 3차원 humanoid를 포함하는 문제군에서 PPO를 학습시킨다. 여기서 로봇은 단순히 앞으로 달리는 것뿐 아니라, 목표 방향으로 조향하고, 바닥에 넘어졌을 때 다시 일어나야 하며, 경우에 따라서는 큐브에 맞으면서도 움직여야 한다. 논문이 다루는 세 과제는 다음과 같다.

1. **RoboschoolHumanoid**: 전방 보행만 수행한다.
2. **RoboschoolHumanoidFlagrun**: 목표 위치가 200 timestep마다 또는 목표에 도달할 때마다 무작위로 바뀐다.
3. **RoboschoolHumanoidFlagrunHarder**: 로봇이 큐브에 맞으면서 넘어질 수 있고, 다시 일어나 목표를 향해야 한다.

논문은 이 세 과제의 학습 곡선을 Figure 4에, 학습된 정책의 정지 프레임을 Figure 5에 제시한다. 하이퍼파라미터는 Appendix A의 Table 4에 제공된다. 또한 동시기(concurrent work)로 Heess et al.이 PPO의 adaptive KL 변형을 사용하여 3D 로봇 locomotion policy를 학습했다는 점도 언급한다.

#### Figure 4 해설

> Figure 4 삽입

Figure 4는 RoboschoolHumanoid-v0, RoboschoolHumanoidFlagrun-v0, RoboschoolHumanoidFlagrunHarder-v0에 대한 학습 곡선을 보여준다. 캡션은 이를 “Roboschool을 사용하는 3D humanoid control task에 대한 PPO 학습 곡선”이라고 요약한다.

세 과제 모두에서 곡선은 시간이 지남에 따라 유의미하게 상승하며, 특히 Flagrun 계열 과제에서는 더 긴 학습 구간 동안 개선이 지속된다. 이는 PPO가 단순한 정방향 보행을 넘어, 목표 전환과 외란 대응까지 포함하는 고난도 제어 과제에서도 학습을 이어갈 수 있음을 보여준다.

#### Figure 5 해설

> Figure 5 삽입

Figure 5는 RoboschoolHumanoidFlagrun에서 학습된 정책의 정지 프레임을 제시한다. 캡션에 따르면 첫 여섯 프레임에서 로봇은 목표를 향해 달리고 있으며, 이후 목표 위치가 무작위로 바뀌자 방향을 전환하여 새로운 목표를 향해 달린다. 즉, 그림은 이 정책이 단순히 보행 주기를 생성하는 것이 아니라, **목표 변화에 따른 조향 행동까지 학습했음**을 시각적으로 보여준다.

### 6.4. Comparison to Other Algorithms on the Atari Domain

연속제어 실험에 이어, 저자들은 PPO를 **Arcade Learning Environment(ALE)** 벤치마크에서도 평가한다. 비교 대상은 잘 튜닝된 **A2C**와 **ACER**이다. 세 알고리즘 모두에 대해 Mnih et al.에서 사용한 것과 동일한 policy network architecture를 사용한다. PPO의 하이퍼파라미터는 Appendix A의 Table 5에 주어지며, A2C와 ACER의 경우에는 이 벤치마크에서의 성능을 최대화하도록 튜닝된 하이퍼파라미터를 사용했다.

논문은 Appendix B에 49개 게임 전체의 결과표와 학습 곡선을 제공한다고 말한다. Atari 실험에서는 두 가지 평가 지표를 사용한다.

1. **전체 학습 구간에 걸친 episode 평균 보상**  
   이는 빠른 학습을 선호하는 지표이다.
2. **학습 마지막 100개 episode의 평균 보상**  
   이는 최종 성능을 선호하는 지표이다.

Table 2는 세 trial에 대해 이 지표를 평균한 뒤, 각 게임에서 어떤 알고리즘이 “이겼는지”를 집계한 결과이다.

#### Table 2. Atari 게임 승수 비교

> Table 2 삽입

표 2는 두 지표가 서로 다른 측면을 측정한다는 점을 명확히 보여준다. 전체 학습 구간 평균에서는 PPO가 30개 게임에서 승리하여 가장 우수하며, 이는 **sample complexity**, 즉 학습 속도 측면의 이점을 시사한다. 반면 마지막 100개 episode 기준의 최종 성능에서는 ACER가 28개 게임으로 가장 많은 승리를 기록한다. 저자들이 서론에서 “Atari에서는 A2C보다 크게 우수하고 ACER와 유사한 수준이나 더 단순하다”고 말한 대목은 바로 이 표에 대응된다. 즉, PPO는 최종 성능만 놓고 보면 ACER와 비슷한 경쟁력을 보이면서, 학습 속도 및 단순성 측면에서 더 유리한 위치를 점한다.

## 7. Conclusion

결론에서 저자들은 PPO를 **각 정책 업데이트마다 여러 epoch의 stochastic gradient ascent를 수행하는 policy optimization 방법군**으로 정리한다. 이어 이 방법군이 **trust-region 계열 방법의 안정성과 신뢰성을 유지하면서도**, 구현은 훨씬 단순하다고 주장한다.

구체적으로 논문은 PPO가 다음과 같은 장점을 갖는다고 요약한다.

- vanilla policy gradient 구현에 비해 **몇 줄의 코드 변경만으로도 적용 가능하다**.
- 정책과 가치함수가 파라미터를 공유하는 경우처럼, **TRPO보다 더 일반적인 설정에 적용 가능하다**.
- 전체적으로 **더 나은 성능**을 보인다.

이 결론은 논문 전반의 구조와도 정확히 맞물린다. 2절과 3절은 왜 보수적 업데이트가 필요한지 이론과 직관을 제공하고, 6절은 clipping 기반 PPO가 실제 벤치마크에서 강한 성능을 낸다는 점을 실험적으로 뒷받침한다.

---

## 8. Acknowledgements

저자들은 Rocky Duan, Peter Chen, 그리고 OpenAI의 다른 동료들에게 유익한 의견에 대해 감사를 표한다.

---

## References

논문 말미의 참고문헌은 다음과 같다.

- **[Bel+15]** M. Bellemare, Y. Naddaf, J. Veness, and M. Bowling. “The arcade learning environment: An evaluation platform for general agents”. In: *Twenty-Fourth International Joint Conference on Artificial Intelligence*. 2015.
- **[Bro+16]** G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba. “OpenAI Gym”. In: *arXiv preprint arXiv:1606.01540* (2016).
- **[Dua+16]** Y. Duan, X. Chen, R. Houthooft, J. Schulman, and P. Abbeel. “Benchmarking Deep Reinforcement Learning for Continuous Control”. In: *arXiv preprint arXiv:1604.06778* (2016).
- **[Hee+17]** N. Heess, S. Sriram, J. Lemmon, J. Merel, G. Wayne, Y. Tassa, T. Erez, Z. Wang, A. Eslami, M. Riedmiller, et al. “Emergence of Locomotion Behaviours in Rich Environments”. In: *arXiv preprint arXiv:1707.02286* (2017).
- **[KL02]** S. Kakade and J. Langford. “Approximately optimal approximate reinforcement learning”. In: *ICML*. Vol. 2. 2002, pp. 267–274.
- **[KB14]** D. Kingma and J. Ba. “Adam: A method for stochastic optimization”. In: *arXiv preprint arXiv:1412.6980* (2014).
- **[Mni+15]** V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al. “Human-level control through deep reinforcement learning”. In: *Nature* 518.7540 (2015), pp. 529–533.
- **[Mni+16]** V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. P. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu. “Asynchronous methods for deep reinforcement learning”. In: *arXiv preprint arXiv:1602.01783* (2016).
- **[Sch+15a]** J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel. “High-dimensional continuous control using generalized advantage estimation”. In: *arXiv preprint arXiv:1506.02438* (2015).
- **[Sch+15b]** J. Schulman, S. Levine, P. Moritz, M. I. Jordan, and P. Abbeel. “Trust region policy optimization”. In: *CoRR, abs/1502.05477* (2015).
- **[SL06]** I. Szita and A. Lőrincz. “Learning Tetris using the noisy cross-entropy method”. In: *Neural Computation* 18.12 (2006), pp. 2936–2941.
- **[TET12]** E. Todorov, T. Erez, and Y. Tassa. “MuJoCo: A physics engine for model-based control”. In: *Intelligent Robots and Systems (IROS), 2012 IEEE/RSJ International Conference on*. IEEE. 2012, pp. 5026–5033.
- **[Wan+16]** Z. Wang, V. Bapst, N. Heess, V. Mnih, R. Munos, K. Kavukcuoglu, and N. de Freitas. “Sample Efficient Actor-Critic with Experience Replay”. In: *arXiv preprint arXiv:1611.01224* (2016).
- **[Wil92]** R. J. Williams. “Simple statistical gradient-following algorithms for connectionist reinforcement learning”. In: *Machine Learning* 8.3–4 (1992), pp. 229–256.

---

## Appendix A. Hyperparameters

부록 A는 PPO 실험에 사용한 하이퍼파라미터를 영역별로 정리한다. 본문에서 간략히 언급되었던 수치들이 이 부록에서 명시적으로 고정된다.

### Table 3. MuJoCo 1M timestep benchmark 하이퍼파라미터

> Table 3 삽입

Table 3는 6.1절 및 6.2절의 MuJoCo 연속제어 실험에서 사용한 PPO 하이퍼파라미터를 제공한다. Horizon이 2048로 비교적 길고, 각 iteration에서 10 epoch의 최적화가 수행되며, minibatch는 64로 설정된다.

### Table 4. Roboschool humanoid 실험 하이퍼파라미터

> Table 4 삽입

Table 4의 캡션은 Adam stepsize가 **KL divergence의 target value에 따라 조정되었다**고 명시한다. 즉, 이 실험은 고정 learning rate보다 목표 KL을 기준으로 한 적응적 조절을 사용한다. 또한 locomotion과 flagrun 계열에서 actor 수가 다르며, action distribution의 log standard deviation을 선형적으로 annealing한다는 점도 중요한 설정이다.

### Table 5. Atari 실험 하이퍼파라미터

> Table 5 삽입

Table 5의 캡션은 \(\alpha\)가 학습 전 구간에 걸쳐 **1에서 0으로 선형 감쇠(linearly annealed)**된다고 밝힌다. 따라서 Atari 실험에서는 learning rate와 clipping parameter가 모두 시간에 따라 함께 줄어든다.

## Appendix B. Performance on More Atari Games

부록 B는 49개 Atari 게임에 대한 상세 결과를 제시한다. 본문은 “PPO against A2C on a larger collection of 49 Atari games”라고 서술하지만, 실제 Figure 6의 범례와 Table 6에는 **A2C, ACER, PPO** 세 알고리즘이 모두 포함되어 있다. Figure 6은 세 random seed 각각의 학습 곡선을, Table 6은 평균 성능을 제공한다.

### Figure 6 해설

> Figure 6 삽입

Figure 6은 당시 OpenAI Gym에 포함되어 있던 49개 Atari 게임 전부에 대해 학습 곡선을 나열한다. 캡션은 PPO와 A2C의 비교라고 적고 있으나, 그림의 범례에는 ACER도 함께 제시된다. 이 그림은 게임마다 알고리즘 간 우열이 다르다는 점을 보여준다. 일부 게임에서는 PPO가 빠르게 상위 성능에 도달하고, 다른 게임에서는 ACER가 더 높은 최종 점수에 도달한다. 본문의 Table 2가 aggregate win count를 제공했다면, Figure 6은 그러한 집계 뒤에 있는 **게임별 학습 양상**을 시각적으로 드러낸다.

### Table 6. 49개 Atari 게임의 mean final score (last 100 episodes)

> Table 6 삽입

Table 6의 캡션은 이 표가 **40M game frames(=10M timesteps) 학습 이후 마지막 100개 episode의 평균 점수**를 제시한다고 명시한다.

표 6를 보면, Atari 도메인에서는 우열이 게임별로 상당히 다르다. 예를 들어 BattleZone, Enduro, Freeway, Gravitar, Jamesbond, Kangaroo, MontezumaRevenge, Robotank, Tennis, TimePilot, WizardOfWor, Zaxxon 등에서는 PPO가 가장 높은 최종 점수를 기록한다. 반면 DemonAttack, Gopher, RoadRunner, VideoPinball 등에서는 ACER가 더 높은 수치를 보인다. Venture는 세 알고리즘이 모두 0.0으로 동일하다. 이러한 세부 결과는 본문 Table 2에서 제시된 승수 집계가 단순 평균 이상의 복합적 양상을 요약한 것임을 보여준다.

부록 B 전체를 종합하면, PPO는 A2C에 비해 명백히 강한 경쟁력을 보이며, ACER와는 게임과 지표에 따라 우열이 갈린다. 이는 본문 6.4절의 결론, 곧 “PPO는 Atari에서 A2C보다 현저히 우수하고 ACER와 비슷한 수준이지만 훨씬 단순하다”는 평가와 정합적이다.
