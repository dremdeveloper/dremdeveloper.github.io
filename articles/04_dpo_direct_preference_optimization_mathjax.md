---
title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
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

- DPO는 RLHF에서 쓰이던 **보상모델 학습과 PPO 단계를 하나의 직접 최적화 문제로 바꾼 방법**이다.
- 선호 데이터만 있으면 chosen / rejected 응답 쌍을 이용해 **바로 정책을 업데이트**할 수 있다는 점이 가장 큰 장점이다.
- 이론적으로는 **KL 제약이 있는 보상 최대화 문제와 직접 분류형 목적식이 연결된다**는 점을 보인다.
- 구조가 단순해 실무 적용이 쉽지만, 여전히 **선호 데이터 품질과 reference model 선택**의 영향을 크게 받는다.

# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

## 확장 해설본
## 문헌 정보

- **논문명**: *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*
- **저자**: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn
- **소속**: Stanford University, CZ Biohub
- **발표**: NeurIPS 2023
- **버전**: arXiv:2305.18290v3 (2024-07-29)
- **비고**: 원문 각주에는 `Equal contribution; more junior authors listed earlier.`가 명시되어 있다.

본 문서는 위 논문의 **초록, 본문, 그림·표 캡션, 각주, 감사의 글, 참고문헌, 저자 기여, 부록 A–D**를 순서대로 재구성한 확장 해설본이다. 목표는 축약 요약이 아니라, 원문을 따라가면서 각 정의, 수식, 실험 설계, 해석, 부록 유도까지 하나의 연속적인 한국어 문서로 읽히도록 정리하는 데 있다. 별도의 응용 지침이나 프로젝트 운영 규칙은 포함하지 않았으며, 설명의 기준은 모두 원문 텍스트이다.

---

## Abstract

논문의 출발점은 대규모 비지도 언어모델이 방대한 지식과 일정 수준의 추론 능력을 획득하더라도, **원하는 방향으로 정밀하게 행동을 제어하는 문제**는 별도로 남아 있다는 진단이다. 원문은 이러한 한계를 “completely unsupervised nature of their training”에서 비롯되는 것으로 정리한다. 즉, 사전학습은 광범위한 통계적 패턴을 학습하게 하지만, 그 패턴들 가운데 실제로 배치하고 싶은 응답 양식만을 안정적으로 선택하게 만들지는 못한다.

이를 보완하기 위해 기존 방법들은 모델이 생성한 여러 응답의 상대적 품질에 대해 인간이 선호 라벨을 부여하고, 그 선호를 따르도록 언어모델을 미세조정해 왔다. 원문은 이 대표적 계열을 RLHF(reinforcement learning from human feedback)로 묶는다. 그러나 RLHF는 먼저 인간 선호를 반영하는 보상모델을 적합하고, 이어서 그 보상을 최대화하되 원래 모델에서 지나치게 벗어나지 않도록 강화학습으로 정책을 조정하는 **2단계 절차**이기 때문에, 복잡하고 불안정한 경향이 있다고 요약한다.

이에 대해 논문은 RLHF의 보상모델을 다른 방식으로 파라미터화하면, 그 보상에 대응하는 최적 정책을 **닫힌형태(closed form)**로 쓸 수 있음을 보인다. 이 관찰을 이용하면 통상적인 RLHF 문제를 강화학습이 아니라 **단순한 분류 손실**로 풀 수 있다. 논문은 이 알고리즘을 Direct Preference Optimization(DPO)라고 부르며, 샘플링 루프와 광범위한 하이퍼파라미터 튜닝 부담을 크게 줄이면서도 계산량이 가볍고 안정적이라고 주장한다.

초록의 마지막 문장은 실험적 주장이다. DPO는 인간 선호에 맞춘 언어모델 정렬에서 기존 방법과 동등하거나 더 나은 성능을 보였고, 특히 감성 제어(sentiment control)에서는 PPO 기반 RLHF보다 우수했으며, 요약과 단일 턴 대화(single-turn dialogue)에서도 응답 품질을 유지하거나 개선하면서 구현과 학습은 훨씬 단순했다고 정리된다. 이 진술은 이후 실험 절에서 세 과제—감성 제어, TL;DR 요약, Anthropic-HH 대화—를 통해 구체화된다.

---

## 1. Introduction

서론은 먼저 대규모 비지도 언어모델의 능력과 한계를 동시에 제시한다. 언어모델은 매우 큰 데이터셋을 통해 놀라운 수준의 지식과 일반화를 획득하지만, 그 데이터는 여러 사람의 목표와 우선순위, 숙련도가 뒤섞인 산물이다. 따라서 모델이 학습한 행동 중에는 **알고 있어야 하지만 따라 해서는 안 되는 것**, 혹은 **드물지만 더 바람직한 패턴을 우선적으로 선택해야 하는 것**이 공존한다.

원문은 두 개의 예시를 든다. 첫째, 코딩 보조 모델은 흔한 프로그래밍 실수를 이해해야 수정할 수 있지만, 실제 코드 생성 단계에서는 훈련 데이터에 드물게 포함된 고품질 코딩 능력 쪽으로 편향되기를 바란다. 둘째, 어떤 사회적 오개념이 전체 인구의 절반에게 널리 퍼져 있다고 해도, 모델이 그 오개념의 존재를 아는 것과 질의의 절반에서 그것을 참으로 답하는 것은 전혀 다른 문제다. 이 두 예시는 모두 **지식의 폭**과 **행동 선택의 정렬**을 구분해야 한다는 논지로 수렴한다. (Section 1)

바로 이 지점에서 논문은 기존 RLHF류 접근을 소환한다. 인간이 선호하는 응답 집합을 별도로 수집하고, 그 선호와 잘 맞는 모델을 학습하는 것이 현재 가장 널리 쓰이는 정렬 경로라는 것이다. 원문은 특히 RLHF/RLAIF 계열이 실제로 유용하고 안전한 행동을 주입하는 데 큰 성과를 냈음을 인정한다. 다만 이러한 파이프라인은 여러 언어모델을 학습해야 하고, 정책 샘플링을 학습 루프 안에서 반복 수행해야 하므로 지도학습보다 훨씬 무겁고 복잡하다.

이 서론의 핵심 주장은, 기존 방법들이 사용하는 RL 기반 목적함수가 사실은 **간단한 binary cross-entropy objective로 정확히 최적화될 수 있다**는 점이다. 다시 말해 논문은 RLHF를 “근사적으로 대체하는 휴리스틱”을 제안하는 것이 아니라, RLHF가 풀고자 했던 제약된 보상 최대화 문제를 다른 변수 표현으로 옮겨 **정확한 분류 문제로 환원**하겠다고 선언한다.

### Figure 1 해설

> Figure 1 삽입

Figure 1은 논문의 전체 메시지를 도식화한 첫 번째 그림이다. 기존 방법은 `(i) 프롬프트와 선호쌍으로 reward model을 학습하고, (ii) 그 learned reward를 최대화하는 policy를 RL로 탐색`한다. DPO는 이 중간 단계를 분리하지 않는다. 그림 캡션이 말하듯, DPO는 “simple classification objective”를 통해 **정책을 직접 최적화**하고, 그 과정에서 암묵적 보상모델(implicit reward model)을 적합한다. 이 도식은 이후 Section 4에서 전개될 “reward function ↔ optimal policy” 매핑을 시각적으로 예고한다. (Figure 1)

서론 말미에서 논문은 자기 기여를 직접 정리한다. 주된 공헌은 선호 데이터로부터 언어모델을 학습하는 **RL-free 알고리즘 DPO**를 제안한 것이며, 감성 제어, 요약, 대화 과제에서 PPO 기반 RLHF를 포함한 기존 방법과 비슷하거나 더 나은 성능을 보였다는 것이다. 여기서 중요한 점은, 서론 단계에서는 이 결론을 선언하지만, 실제 논거는 뒤의 Section 4–6에서 순차적으로 제공된다는 점이다.

## 2. Related Work

관련연구 절은 DPO를 세 갈래의 문헌 맥락에 배치한다.

첫 번째 갈래는 **instruction tuning**이다. 원문은 대규모 자기지도 언어모델이 zero-shot 또는 few-shot으로도 작업을 수행할 수 있지만, 지시문과 인간 작성 completion으로 미세조정하면 다운스트림 성능과 사용자 의도 정렬이 현저히 좋아진다는 선행연구를 정리한다. 즉, 사전학습 후에 추가적인 목표 지향형 미세조정이 필요하다는 점은 이미 널리 받아들여진 전제다. (Section 2)

두 번째 갈래는 **인간 선호 데이터에 의한 미세조정**이다. 번역, 요약, 스토리 생성, instruction following 등에서 상대적 인간 판단이 전문가 데모보다 수집하기 쉬운 경우가 많기 때문에, 이후 연구들은 응답 쌍에 대한 선호 데이터셋을 활용해 언어모델을 개선해 왔다. 이 표준 접근은 대체로 Bradley–Terry 같은 선호모델 아래 reward function을 먼저 학습한 뒤, REINFORCE나 PPO와 같은 RL 알고리즘으로 정책을 그 reward에 맞춰 조정하는 형태를 갖는다. DPO는 바로 이 흐름을 겨냥한다.

세 번째 갈래는 **언어 바깥의 preference learning**이다. 논문은 contextual dueling bandit(CDB)과 preference-based RL(PbRL)을 언급한다. CDB는 보상이 아니라 행동들 사이의 선호 또는 랭킹을 관찰하며, 이론적으로는 절대적 optimal policy 대신 von Neumann winner 같은 개념을 사용한다. PbRL 역시 이진 선호를 미지의 scoring function에서 파생된 것으로 보고, 보통은 그 scoring function—즉 latent reward—를 먼저 추정한 뒤 최적화한다. 논문은 이와 달리 DPO가 **single-stage policy learning**이라는 점을 분명히 한다. 즉, 보상 추정과 정책 최적화를 분리하지 않고, 선호를 만족시키는 정책을 직접 학습한다는 것이다.

또 하나의 인접 문헌으로는, 이미 instruction following에 맞춰 튜닝된 LLM을 이용해 합성 선호 데이터를 생성하는 계열(RLAIF, Constitutional AI 등)이 언급된다. 논문은 인간이 약한 감독만 제공하고 AI가 선호를 확장 생성하는 흐름도 정렬 파이프라인의 일부로 이해한다. 다만 DPO의 기여는 라벨을 누가 붙이느냐보다는 **그 라벨로 policy를 어떻게 최적화하느냐**에 있다.

---

## 3. Preliminaries

이 절은 DPO가 대체하려는 기존 RLHF 파이프라인을 엄밀하게 정의한다. 핵심 표기법은 다음과 같다.

| 기호 | 의미 |
|---|---|
| \(x\) | 프롬프트(prompt) |
| \(y\) | 응답 또는 completion |
| \(y_w\) | 선호된 응답(preferred / winner) |
| \(y_l\) | 비선호 응답(dispreferred / loser) |
| \(\pi^{\mathrm{SFT}}\) | 지도 미세조정으로 얻은 초기 정책 |
| \(\pi_{\mathrm{ref}}\) | KL 제약의 기준이 되는 reference policy |
| \(r^*(x,y)\) | 관측 불가능한 잠재 보상 함수 |
| \(r_\phi(x,y)\) | 학습된 reward model |
| \(\beta\) | reference에서 벗어나는 정도를 제어하는 계수 |

논문은 RLHF가 대체로 세 단계로 구성된다고 정리한다. (Section 3)

### 3.1 SFT 단계

첫 단계는 고품질 데이터로 사전학습 언어모델을 지도 미세조정하여 \(\pi^{\mathrm{SFT}}\)를 만드는 것이다. 대화, 요약, 코딩 등 다운스트림 과제에 따라 이 모델이 이후 선호 수집과 정렬의 출발점이 된다.

### 3.2 Reward Modeling 단계

둘째 단계에서는 SFT 모델에 프롬프트 \(x\)를 주어 여러 응답을 생성하고, 그 중 두 응답 \((y_1, y_2)\)를 인간 라벨러가 비교하여 어느 쪽을 더 선호하는지 표시한다. 선호된 응답을 \(y_w\), 선호되지 않은 응답을 \(y_l\)로 표기한다. 원문은 이 선호가 어떤 잠재 reward function \(r^*(x,y)\)에서 생성된다고 가정한다.

이때 가장 널리 쓰이는 선호모델이 Bradley–Terry 모델이다.

$$
p^*(y_1 \succ y_2 \mid x)
=
\frac{\exp(r^*(x,y_1))}{\exp(r^*(x,y_1)) + \exp(r^*(x,y_2))}
\tag{1}
$$

이 식이 뜻하는 바는 단순하다. 특정 프롬프트 \(x\) 아래에서 응답 \(y_1\)의 잠재 보상이 더 크면, 인간이 \(y_1\)을 \(y_2\)보다 더 자주 선택한다. 중요한 점은 **보상의 절대값이 아니라 차이**가 선호확률을 좌우한다는 사실이다.

정적 비교 데이터셋 \(D = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^N\)가 주어졌다고 할 때, reward model은 다음과 같은 최대우도 목적을 통해 학습된다.

$$
L_R(r_\phi, D)
=
-\mathbb{E}_{(x,y_w,y_l)\sim D}\big[\log \sigma(r_\phi(x,y_w)-r_\phi(x,y_l))\big]
\tag{2}
$$

여기서 \(\sigma\)는 logistic sigmoid이다. 본질적으로 이는 “선호된 응답의 점수가 비선호 응답보다 높아야 한다”는 **이진 분류 손실**이다. 원문은 실제 LM 맥락에서 reward model을 SFT 모델로 초기화한 뒤 마지막 transformer 레이어 위에 단일 스칼라 보상을 출력하는 선형 헤드를 올리는 방식이 자주 사용된다고 적는다. 또한 reward variance를 줄이기 위해 보상을 정규화하는 선행관행도 언급한다.

### 3.3 RL Fine-Tuning 단계

셋째 단계는 학습된 reward model을 사용해 언어모델 정책을 미세조정하는 단계다. 원문은 표준적인 RLHF 목적을 다음과 같이 제시한다.

$$
\max_{\pi_\theta}
\;\mathbb{E}_{x\sim D,\; y\sim \pi_\theta(\cdot\mid x)}[r_\phi(x,y)]
- \beta D_{\mathrm{KL}}\bigl(\pi_\theta(y\mid x)\;\|\;\pi_{\mathrm{ref}}(y\mid x)\bigr)
\tag{3}
$$

첫 번째 항은 보상이 높은 응답을 더 자주 생성하도록 유도하고, 두 번째 항은 정책이 reference policy에서 지나치게 벗어나지 않게 만든다. 이 KL 제약은 reward model이 상대적으로 신뢰할 수 있는 분포 근방에 정책을 묶어 두는 역할을 하며, 동시에 다양성 유지와 mode collapse 방지에도 기여한다.

문제는 언어 생성이 이산적이기 때문에 위 목적이 직접 미분 가능하지 않다는 점이다. 그래서 실무적으로는 보상에 KL 벌점을 합친 형태의 surrogate reward를 만든 뒤 PPO로 정책을 최적화하는 것이 관례가 되어 왔다. DPO는 바로 이 지점에서, RL 단계 자체를 다른 형태의 supervised objective로 대체할 수 있는지를 묻는다.

---

## 4. Direct Preference Optimization

Section 4는 논문의 방법론 핵심이다. 여기서 저자들은 “보상함수에 대한 최적 정책”을 해석적으로 표현하고, 그 표현을 다시 뒤집어 **정책 자체를 보상모델의 새로운 파라미터화**로 사용한다.

### 4.1 KL-제약 보상 최대화의 최적해

원문은 먼저 Section 3의 RLHF 목적식(Eq. 3)에서 출발한다. 기존 보상 함수 \(r(x,y)\)가 주어졌다고 하면, KL 제약이 있는 보상 최대화 문제의 최적 정책은 다음 형태를 갖는다.

$$
\pi_r(y\mid x)
=
\frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y\mid x)
\exp\!\left(\frac{1}{\beta}r(x,y)\right)
\tag{4}
$$

여기서

$$
Z(x)=\sum_y \pi_{\mathrm{ref}}(y\mid x)
\exp\!\left(\frac{1}{\beta}r(x,y)\right)
$$

는 partition function이다. 이 식은 직관적으로, reference policy 위에 reward가 높은 응답일수록 지수적으로 큰 가중을 주고, 마지막에 전체를 정규화한 분포라고 볼 수 있다. \(\beta\)가 작을수록 reward 차이가 더 날카롭게 반영되고, \(\beta\)가 클수록 reference에 더 가까운 정책이 된다. 원문은 자세한 유도를 Appendix A.1로 보낸다.

그러나 Eq. (4)는 그대로 구현하기 어렵다. \(Z(x)\)는 가능한 모든 completion에 대한 합을 포함하며, 언어모델의 출력공간에서는 이 값을 정확히 계산하는 것이 사실상 불가능하다. 따라서 RLHF에서는 통상 샘플 기반 RL로 이 문제를 우회했다.

### 4.2 보상을 정책으로 다시 쓰기

저자들은 Eq. (4)를 로그화하여 다음 관계를 얻는다.

$$
r(x,y)
=
\beta \log \frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+ \beta \log Z(x)
\tag{5}
$$

이 식은 매우 중요하다. 보상이란 결국 **특정 응답을 reference보다 얼마나 더 선호하도록 정책을 기울였는가**를 나타내는 로그비(log ratio)로 표현될 수 있기 때문이다. 마지막의 \(\beta \log Z(x)\)는 프롬프트 \(x\)에만 의존하는 항이다.

### 4.3 Bradley–Terry 모델에서의 상쇄(cancellation)

Bradley–Terry 모델은 두 응답의 보상 차이만 사용한다. 따라서 \(r(x,y_1)-r(x,y_2)\)를 계산할 때 Eq. (5)의 \(\beta \log Z(x)\)는 소거된다. 이 점을 이용하면 잠재 보상 \(r^*\)에 대한 인간 선호확률을 **최적 정책 \(\pi^*\)와 reference policy만으로** 다시 쓸 수 있다.

$$
p^*(y_1 \succ y_2 \mid x)
=
\frac{1}{1+\exp\left(\beta\log\frac{\pi^*(y_2\mid x)}{\pi_{\mathrm{ref}}(y_2\mid x)}-\beta\log\frac{\pi^*(y_1\mid x)}{\pi_{\mathrm{ref}}(y_1\mid x)}\right)}
\tag{6}
$$

즉, 인간 선호는 결국 “reference 대비 정책이 winner를 loser보다 얼마나 더 밀어 주는가”로 재표현된다. 원문은 이 유도를 Appendix A.2에, 보다 일반적인 Plackett–Luce 랭킹 모델에 대한 확장을 Appendix A.3에 배치한다.

### 4.4 DPO 목적함수

이제 reward model을 따로 두지 않고도, 파라미터화된 정책 \(\pi_\theta\)에 대한 최대우도 목적을 세울 수 있다. 논문이 제안하는 DPO 손실은 다음과 같다.

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta; \pi_{\mathrm{ref}})
=
-\mathbb{E}_{(x,y_w,y_l)\sim D}
\left[
\log \sigma\!\left(
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right)
\right]
\tag{7}
$$

형태상으로는 로지스틱 이진분류와 동일하다. 그러나 입력 변수는 일반적인 분류 모델의 점수가 아니라, **정책이 reference에 대해 보이는 상대 로그확률 차이**다. 이 목적식은 “선호된 응답의 상대 로그확률을 더 높이고, 비선호 응답의 상대 로그확률을 더 낮추라”는 방향을 직접 구현한다.

저자들이 강조하는 지점은 다음과 같다. Eq. (7)을 최소화하는 과정은 표면적으로는 policy fitting처럼 보이지만, 실제로는 Eq. (5)에 의해 정의되는 **암묵적 reward**를 적합하고 있는 셈이며, 그 reward에 대응하는 optimal policy가 바로 \(\pi_\theta\)이다. 이 때문에 논문 제목의 문구—“Your Language Model is Secretly a Reward Model”—가 성립한다.

### 4.5 DPO gradient의 의미

Section 4는 DPO 업데이트가 실제로 무엇을 하는지 gradient 해석까지 제시한다. 원문은 다음과 같은 형태를 도출한다.

$$
\nabla_\theta \mathcal{L}_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}})
=
-\beta\,\mathbb{E}_{(x,y_w,y_l)\sim D}
\Big[
\sigma\big(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\big)
\big(\nabla_\theta\log\pi_\theta(y_w\mid x)-\nabla_\theta\log\pi_\theta(y_l\mid x)\big)
\Big]
$$

여기서 암묵적 reward는

$$
\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
$$

로 정의된다. 괄호 안의 항은 winner의 로그확률을 올리고 loser의 로그확률을 내리는 표준적인 쌍대비교 업데이트다. 중요한 차이는 앞에 곱해진

$$
\sigma\big(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\big)
$$

라는 **예시별 동적 가중치**다. 현재 모델이 loser를 winner보다 더 높게 보고 있을수록 이 값은 커지고, 이미 올바른 순서를 잘 맞추고 있을수록 값이 작아진다. 원문은 바로 이 동적 가중이 단순 확률비 목적의 퇴행(degeneration)을 막는 핵심이라고 설명하며, 가중치 없는 나이브 목적이 실제로 모델 붕괴를 일으키는 사례를 Appendix Table 3에서 제시한다.

### 4.6 알고리즘 개요와 reference policy 선택

DPO 파이프라인은 두 단계로 정리된다. 첫째, reference policy \(\pi_{\mathrm{ref}}\)에서 completion 쌍을 샘플링하고 인간 선호로 라벨링하여 오프라인 데이터셋 \(D=\{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^N\)를 구성한다. 둘째, 주어진 \(\beta\)와 reference policy에 대해 Eq. (7)을 최소화하도록 \(\pi_\theta\)를 학습한다. (Section 4)

원문은 실무적 상황도 덧붙인다. 공개 선호 데이터셋은 대개 어떤 SFT 정책에서 샘플된 것이므로, 가능하면 \(\pi_{\mathrm{ref}}=\pi^{\mathrm{SFT}}\)로 두는 것이 자연스럽다. 만약 그런 SFT 정책이 없다면, 선호된 completion들만 이용해 다음 maximum likelihood objective를 푼 모델로 reference를 초기화할 수 있다.

$$
\pi_{\mathrm{ref}}
=
\arg\max_\pi \;\mathbb{E}_{x,y_w\sim D}[\log \pi(y_w\mid x)]
$$

논문은 이 절차가 “진짜 reference distribution”과 DPO 학습에 사용되는 \(\pi_{\mathrm{ref}}\) 사이의 distribution shift를 줄이는 데 도움이 된다고 설명한다.

---

## 5. Theoretical Analysis of DPO

Section 5의 목적은 두 가지다. 첫째, DPO가 단순한 학습 요령이 아니라 **reward model의 표현 자체를 policy 공간으로 옮긴 정식한 재파라미터화**임을 보이는 것이다. 둘째, 이 관점에서 기존 actor-critic 계열 RLHF, 특히 PPO의 불안정성을 다시 해석하는 것이다.

### 5.1 Your Language Model Is Secretly a Reward Model

이 소절의 첫 문장은 DPO의 의미를 집약한다. DPO는 명시적 reward를 따로 적합하지도 않고, RL로 policy를 따로 학습하지도 않으며, **하나의 최대우도 목적**만으로 두 단계를 동시에 우회한다. 이는 Section 4에서 얻은 재파라미터화

$$
r^*(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
$$

를 전제로 한다. 즉, 보상을 직접 출력하는 별도 네트워크를 두는 대신, 정책이 reference에 비해 각 응답을 얼마나 더 선호하는지를 보상의 대표 표현으로 본다. (Section 5.1)

#### Definition 1: reward equivalence class

논문은 reward의 식별성 문제를 정식화하기 위해 다음 동치관계를 정의한다.

$$
r(x,y) \sim r'(x,y)
\quad\Leftrightarrow\quad
r(x,y)-r'(x,y)=f(x)
$$

여기서 \(f(x)\)는 응답 \(y\)와 무관하게 프롬프트 \(x\)에만 의존하는 함수다. 다시 말해, 두 reward가 prompt별 상수항만큼 차이 난다면 두 reward는 같은 equivalence class에 속한다. 이 정의는 Bradley–Terry나 Plackett–Luce 류 선호모델이 본질적으로 **응답 간 reward 차이**만을 보기 때문에 자연스럽다. (Definition 1)

#### Lemma 1과 Lemma 2

논문은 곧바로 두 개의 보조정리를 제시한다.

- **Lemma 1**: 같은 equivalence class에 속한 reward 함수들은 Plackett–Luce, 특히 Bradley–Terry 선호 프레임워크 아래에서 동일한 preference distribution을 유도한다.
- **Lemma 2**: 같은 equivalence class에 속한 reward 함수들은 KL-제약 RL 문제에서 동일한 optimal policy를 유도한다.

이 두 정리는 DPO의 핵심 논리를 받쳐 준다. 선호 데이터가 본질적으로 reward의 절대 기준선을 식별하지 못하더라도, 우리가 관심 있는 것은 그 reward가 유도하는 **정책**이며, 그 정책은 equivalence class 수준에서 이미 고정된다는 것이다. 원문은 두 증명을 Appendix A.5로 미룬다.

#### Theorem 1: 재파라미터화의 일반성

Section 5.1의 핵심 정리는 다음과 같이 진술된다.

> Plackett–Luce(특히 Bradley–Terry) 모델과 양립하는 모든 reward equivalence class는, 어떤 모델 \(\pi(y\mid x)\)와 주어진 reference model \(\pi_{\mathrm{ref}}(y\mid x)\)에 대해
>
$$
r(x,y)=\beta\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
$$
>
> 의 형태로 표현될 수 있다.

이 정리는 “DPO가 reward 표현력을 지나치게 제한하는 것 아니냐”는 우려를 정면으로 다룬다. 저자들의 대답은 그렇지 않다는 것이다. reward를 policy의 로그비로 쓰더라도, 선호모델이 구분할 수 있는 equivalence class 전체는 그대로 표현할 수 있다.

이를 위해 논문은 다음과 같은 projection 연산을 정의한다.

$$
f(r;\pi_{\mathrm{ref}},\beta)(x,y)
=
r(x,y)-\beta\log\sum_y \pi_{\mathrm{ref}}(y\mid x)\exp\!\left(\frac{1}{\beta}r(x,y)\right)
\tag{8}
$$

이 연산은 reward에서 partition function의 로그를 빼는 정규화다. 빼는 항은 \(x\)에만 의존하므로 동일한 equivalence class 안에 머문다. 그리고 Eq. (5)를 대입하면, 이 정규화된 reward는 정확히

$$
f(r;\pi_{\mathrm{ref}},\beta)(x,y)=\beta\log\frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
$$

형태가 된다. 따라서 DPO의 재파라미터화는 하나의 equivalence class마다 “대표 reward”를 하나 선택하는 방식으로 이해될 수 있다.

원문은 이를 다시 다음 조건으로 표현한다.

$$
\sum_y \pi_{\mathrm{ref}}(y\mid x)\exp\!\left(\frac{1}{\beta}r(x,y)\right)=1
\tag{9}
$$

Eq. (9)는 그 reward가 유도하는 optimal policy의 partition function을 1로 맞춘 형태이며, 본질적으로 각 class 안에서의 정규화 조건이다. 논문은 이 조건 덕분에 Plackett–Luce 계열의 under-specification을 해치지 않으면서, policy를 해석적으로 다룰 수 있는 형태를 얻는다고 주장한다.

### 5.2 Instability of Actor-Critic Algorithms

두 번째 소절은 PPO 같은 actor-critic 계열 RLHF가 왜 불안정할 수 있는지를 DPO 관점에서 재해석한다. 논문은 KL-제약 RL 문제를 “control as inference” 관점과 연결하고, parameterized policy \(\pi_\theta\)가 reward \(r_\phi\)가 유도하는 optimal policy \(\pi^*\)와의 KL을 최소화하는 문제로 본다.

이 해석을 통해 다음 목적식이 나온다.

$$
\max_{\pi_\theta}
\;\mathbb{E}_{y\sim\pi_\theta(y\mid x)}
\Big[
\underbrace{r_\phi(x,y)-\beta\log\sum_y\pi_{\mathrm{ref}}(y\mid x)\exp\!\left(\frac{1}{\beta}r_\phi(x,y)\right)}_{f(r_\phi,\pi_{\mathrm{ref}},\beta)}
-
\underbrace{\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}}_{\text{KL term}}
\Big]
\tag{10}
$$

여기서 첫 번째 묶음은 reward에 포함된 정규화항을 드러낸 것이고, 두 번째 묶음은 reference 대비 KL 제약이다. 논문의 핵심 진단은 다음과 같다. 이 정규화항은 **최적해 자체를 바꾸지는 않지만**, policy gradient의 분산을 크게 좌우한다. 실제 PPO 구현에서는 이 항을 value function 또는 baseline 형태로 근사해야 하는데, 그 학습이 어렵고 불안정해질 수 있다. 반면 DPO 재파라미터화는 이러한 baseline을 명시적으로 학습할 필요가 없는 reward 표현을 제공한다.

이 소절은 “DPO가 왜 PPO보다 단순한가”를 설명하는 수준을 넘어, “PPO가 왜 더 많은 보정 장치와 튜닝을 필요로 하는가”를 이론적으로 해석한다는 점에서 중요하다.

---

## 6. Experiments

실험 절은 DPO를 세 가지 과제에서 평가한다. 하나는 **ground-truth reward가 있는 통제 실험**이고, 나머지 둘은 **실제 인간 선호 데이터**가 주어진 요약과 대화다. 원문은 먼저 과제, 평가 방법, 비교군을 정리한 뒤, 각 실험 질문에 대응하는 하위 절을 제시한다. (Section 6)

### 6.0 실험 설정: 과제, 평가, 비교군

#### 과제

| 과제 | 입력 \(x\) | 출력 \(y\) | 데이터 및 특징 |
|---|---|---|---|
| Controlled sentiment generation | IMDb 리뷰 prefix | 긍정 감성 completion | 선호쌍을 사람이 아니라 사전학습 감성 분류기로 부여 |
| TL;DR summarization | Reddit 포럼 글 | 요약문 | Reddit TL;DR + Stiennon et al.의 인간 선호 데이터 |
| Single-turn dialogue | 인간 질의 | 도움 되는 단일 응답 | Anthropic Helpful & Harmless, 약 170k dialogues |

감성 제어에서는 IMDb 리뷰 prefix를 입력으로 주고, 긍정 감성의 continuation을 생성하도록 정책을 학습한다. 이 실험은 사람이 아니라 **사전학습 감성 분류기**로 선호쌍을 구성하므로, 알고리즘이 최적화해야 할 진짜 보상 함수에 접근할 수 있다는 장점이 있다.

요약에서는 Reddit TL;DR 데이터셋과 인간 선호 데이터가 사용된다. 대화에서는 Anthropic HH 데이터셋이 사용되며, 각 transcript 끝에 대형 언어모델이 생성한 두 개의 응답과 인간 선호 라벨이 붙어 있다. 이 대화 설정에서는 표준적인 SFT 모델이 미리 준비되어 있지 않으므로, 원문은 preferred completions만으로 off-the-shelf LM을 fine-tune하여 초기 reference를 구성한다고 설명한다.

#### 평가 방법

논문은 과제 유형에 따라 두 가지 평가 방식을 분리한다.

1. **Reward–KL frontier**: 통제 감성 실험에서는 true reward와 reference policy에 대한 KL을 모두 계산할 수 있으므로, 평균 reward와 평균 KL의 trade-off frontier로 알고리즘을 비교한다.
2. **Win rate**: 요약과 대화에서는 ground-truth reward가 없기 때문에, baseline 응답과 head-to-head로 비교한 승률을 사용한다. 이때 GPT-4를 자동 평가자로 사용하고, Section 6.4에서 사람 평가를 통해 그 타당성을 검증한다.

요약의 baseline은 test split의 인간 작성 reference summary이고, 대화의 baseline은 test set에서 preferred response로 표시된 chosen completion이다.

#### 비교군

실험 비교군은 다음과 같다.

| 방법 | 설명 |
|---|---|
| Zero-shot prompting | 요약은 GPT-J, 대화는 Pythia-2.8B 2-shot prompting |
| SFT | 기본 지도 미세조정 모델 |
| Preferred-FT | 선호 데이터에서 winner만 정답으로 사용한 지도학습 모델 |
| Unlikelihood | winner 확률을 키우고 loser 확률을 낮추는 목적 |
| PPO | learned reward를 쓰는 표준 RLHF |
| PPO-GT | 감성 통제 실험에서만 가능한 oracle reward 기반 PPO |
| Best of \(N\) | 여러 샘플을 뽑아 learned reward로 최고점을 선택 |
| DPO | 제안 방법 |

Best of \(N\)은 성능은 강하지만, 쿼리마다 여러 응답을 생성하고 다시 스코어링해야 하므로 테스트 시점 계산량이 매우 크다고 논문은 지적한다.

### Figure 2와 Figure 3의 역할

> Figure 2 삽입

- **Figure 2 왼쪽**: IMDb 감성 제어에서 expected reward와 \(KL(\pi\|\pi_{\mathrm{ref}})\)의 frontier를 나타낸다.
- **Figure 2 오른쪽**: TL;DR 요약에서 sampling temperature에 따른 GPT-4 win rate를 나타낸다.
- **Figure 3 왼쪽**: Anthropic-HH 대화에서 chosen completion 대비 승률을 보여 준다.
- **Figure 3 오른쪽**: DPO의 학습 단계별 성능 추이를 temperature별로 나타낸다.

이 두 그림은 각각 “최적화 효율”과 “실제 선호 데이터에서의 품질 변화”를 시각화한다.

### 6.1 How well can DPO optimize the RLHF objective?

이 소절의 질문은 명확하다. DPO와 PPO가 동일한 KL-제약 보상 최대화 목적을 겨냥한다면, 실제로 어느 쪽이 더 **효율적인 frontier**를 형성하는가?

논문은 IMDb 감성 실험에서 다음과 같이 하이퍼파라미터를 바꿔가며 총 22개 run을 수행한다.

- PPO: target KL \(\in \{3, 6, 9, 12\}\)
- DPO: \(\beta \in \{0.05, 0.1, 1, 5\}\)
- Unlikelihood: \(\alpha \in \{0.05, 0.1, 0.5, 1\}\)
- Preferred-FT: random seed만 변경

각 run은 100 training step마다 평가되며, test prompt에 대해 true reward 평균과 sequence-level KL 평균을 계산한다. 여기서 sequence-level KL은 원문 각주가 명시하듯 **각 시점 KL의 합**이다.

Figure 2 왼쪽의 해석은 분명하다. DPO가 reward–KL 평면에서 가장 우수한 frontier를 형성한다. 원문은 이를 두 가지 의미로 해석한다. 첫째, DPO와 PPO는 동일한 목적을 최적화한다고 주장되지만 실제 계산 효율은 다르며, DPO가 더 유리하다. 둘째, 심지어 PPO가 ground-truth reward를 직접 사용할 수 있는 PPO-GT 환경에서도, DPO가 더 좋은 frontier를 형성한다.

### 6.2 Can DPO scale to real preference datasets?

이 소절에서는 요약과 대화라는 더 현실적인 선호 데이터셋에서 DPO를 평가한다.

#### TL;DR 요약

원문은 먼저 요약 과제에서 ROUGE류 자동 지표가 인간 선호와 약하게 상관할 수 있다는 선행연구를 상기시킨다. 따라서 test split에서 각 알고리즘이 생성한 요약을 인간 작성 reference summary와 비교하여 GPT-4 win rate를 계산한다.

모든 방법은 temperature 0.0부터 1.0까지 sweep되며, 결과는 Figure 2 오른쪽에 제시된다. 핵심 수치는 본문에 직접 명시되어 있다.

- DPO: temperature 0.0에서 약 **61%** win rate
- PPO: 최적 temperature 0.0에서 약 **57%** win rate

또한 DPO는 Best-of-\(N\) baseline보다도 더 높은 최대 win rate를 달성했다고 서술된다. 저자들은 DPO의 \(\beta\)를 의미 있게 튜닝하지 않았으므로, 이 결과가 잠재력을 오히려 과소평가했을 가능성도 언급한다.

Figure 2 오른쪽에서 더 중요한 관찰은 **temperature에 대한 견고성**이다. DPO는 temperature가 변해도 성능이 완만하게 움직이는 반면, PPO는 높은 temperature에서 base GPT-J 수준으로 급격히 떨어질 수 있다. Preferred-FT는 SFT 대비 큰 개선을 보이지 않는다.

#### Anthropic-HH 단일 턴 대화

대화 실험에서는 test split 중 one-step human-assistant interaction 서브셋을 사용한다. baseline은 데이터셋의 chosen completion이며, GPT-4가 각 방법과 chosen completion의 head-to-head 승률을 평가한다.

이 과제에는 미리 학습된 표준 SFT 모델이 없기 때문에, 저자들은 pretrained Pythia-2.8B에서 출발해 preferred completions만으로 먼저 reference 모델을 학습한다. 그런 다음 DPO를 적용한다. 비교군으로는 2-shot prompted Pythia-2.8B, 그리고 Preferred-FT에서 128개 completion을 샘플한 뒤 최고 reward를 고르는 Best of 128이 포함된다. 논문은 이 태스크에서 Best of \(N\)의 성능이 대체로 \(N=64\sim 128\) 수준에서 포화된다고 Appendix Figure 4를 통해 보인다고 말한다.

Figure 3 왼쪽의 메시지는 명료하다. DPO는 Anthropic-HH test set의 chosen completions보다도 개선되는 **유일한 계산 효율적 방법**으로 제시된다. Best of 128은 성능이 높지만 테스트 계산량이 크고, PPO로 학습된 공개 RLHF 모델은 적절한 프롬프트나 temperature를 찾지 못해 base Pythia-2.8B보다 낫게 만들지 못했다고 보고한다.

Figure 3 오른쪽은 학습 단계에 따른 대화 win rate evolution을 보여 준다. DPO는 비교적 빠르게 0.6 수준 안팎의 승률로 수렴하며, temperature 0.7과 1.0 모두에서 비슷한 추이를 보인다. 논문은 이를 “improvement over the dataset labels is fairly stable”하다고 해석한다.

### 6.3 Generalization to a new input distribution

OOD 일반화 실험은 Reddit TL;DR에서 학습한 정책을 CNN/DailyMail 뉴스 기사에 적용하여 수행된다. GPT-4(C) 프롬프트는 기존 요약 평가 프롬프트와 동일하되, “forum post”만 “news article”로 바꾸어 사용한다.

Table 1의 수치는 다음과 같다.

| 알고리즘 | Temp 0 | Temp 0.25 |
|---|---:|---:|
| DPO | 0.36 | 0.31 |
| PPO | 0.26 | 0.23 |

논문은 이 결과를 두 가지로 해석한다. 첫째, 새로운 입력 분포에서도 DPO가 PPO를 유의미하게 앞선다. 둘째, PPO는 추가 unlabeled Reddit TL;DR prompt를 활용하는 반면, DPO는 그런 보조 데이터를 쓰지 않는데도 comparable하거나 더 나은 일반화를 보인다는 점에서 고무적이라는 것이다.

### 6.4 Validating GPT-4 judgments with human judgments

이 절은 GPT-4를 자동 평가자로 사용하는 실험 체계 자체를 검증한다. 논문은 TL;DR 요약 실험의 결과 샘플을 사람에게 다시 비교하게 하여, GPT-4 판단과 인간 판단의 상관 및 합의율을 측정한다.

GPT-4 프롬프트는 두 가지다.

- **GPT-4 (S)**: 어떤 요약이 포럼 글의 가장 중요한 요점을 더 잘 요약하는가를 묻는 단순 프롬프트
- **GPT-4 (C)**: 여기에 더해 “불필요하거나 무관한 세부를 포함하지 말 것, 좋은 요약은 정확하면서 간결해야 한다”는 제약을 넣은 프롬프트

저자들이 이 두 프롬프트를 구분한 이유는, 단순 프롬프트(S)에서 GPT-4가 사람보다 **길고 반복적인 요약을 더 선호하는 경향**을 보였기 때문이다.

비교는 세 가지 매치업으로 구성된다.

1. DPO (temperature 0.25) vs PPO (temperature 0.0)
2. SFT (temperature 0.25) vs PPO (temperature 0.0)
3. PPO-1, 즉 PPO (temperature 1.0) vs PPO (temperature 0.0)

Table 2의 수치는 다음과 같다.

| 비교 대상 | 응답자 수 | GPT-4(S) win% | GPT-4(C) win% | Human win% | GPT-4(S)-Human agree | GPT-4(C)-Human agree | Human-Human agree |
|---|---:|---:|---:|---:|---:|---:|---:|
| DPO | 272 | 47 | 54 | 58 | 70 | 67 | 65 |
| SFT | 122 | 27 | 32 | 43 | 77 | 79 | - |
| PPO-1 | 199 | 13 | 12 | 17 | 86 | 85 | 87 |

원문 해석은 다음과 같다. GPT-4의 판단은 사람 판단과 강한 상관을 보이며, GPT-4–Human agreement는 대체로 Human–Human agreement와 비슷한 수준이다. 또한 (C) 프롬프트가 인간 승률을 더 잘 반영하므로, Section 6.2의 메인 결과는 GPT-4(C) 프롬프트를 사용한다.

---

## 7. Discussion

논문은 Discussion에서 DPO를 “강화학습 없이 선호로부터 언어모델을 학습하는 단순한 패러다임”으로 재정의한다. 여기서 저자들이 강조하는 대조는 다음과 같다. 기존 방법은 선호 학습 문제를 표준 RL 문제로 변환한 뒤 오프더셸프 RL 알고리즘을 적용한다. 반면 DPO는 정책과 보상 사이의 매핑을 이용해, **policy를 직접 선호에 맞추는 cross-entropy objective**를 구성한다. 이 과정에서 강화학습도 필요하지 않고, reward model의 일반성도 잃지 않는다고 주장한다.

또한 저자들은, 하이퍼파라미터를 거의 튜닝하지 않았음에도 DPO가 PPO 기반 RLHF를 포함한 기존 방법과 동등하거나 더 나은 성능을 보였다고 다시 정리한다. 이때 핵심은 “절대적 최고 성능”의 선언보다는, **정렬 파이프라인의 진입장벽을 유의미하게 낮출 수 있다**는 실용적 메시지다.

### 한계와 향후 과제

원문은 Discussion 끝에서 몇 가지 열린 질문을 명시한다.

1. **OOD 일반화**: DPO 정책이 explicit reward를 학습하는 계열과 비교해 분포 밖에서 어떤 일반화 특성을 보이는지 더 체계적 분석이 필요하다.
2. **Unlabeled prompt 활용**: DPO 정책이 스스로 라벨을 생성하는 self-labeling을 통해, PPO처럼 라벨 없는 프롬프트까지 활용할 수 있는지 탐구할 가치가 있다.
3. **Reward over-optimization**: 직접 선호 최적화에서도 과최적화가 어떤 형태로 나타나는지, Figure 3 오른쪽의 약간의 성능 하락이 그 사례인지 조사해야 한다.
4. **스케일링**: 본 논문은 최대 6B 규모까지 평가했으므로, 훨씬 더 큰 최신 모델로의 확장이 남아 있다.
5. **평가 프롬프트 민감도**: GPT-4 win rate가 프롬프트 설계에 영향을 받는 만큼, 자동 평가를 더 견고하게 만드는 방법이 향후 연구 과제다.
6. **다른 모달리티**: DPO는 인간 선호 기반 생성모델 전반으로 확장될 수 있는 가능성이 있다.

논문은 여기서 DPO를 완결된 해법으로 제시하기보다, RLHF를 훨씬 단순한 형태로 다시 쓰는 첫 번째 정식화로 제시한다.

---

## Acknowledgements

감사의 글에서 논문은 다음 지원을 명시한다.

- Eric Mitchell은 Knight-Hennessy Graduate Fellowship의 지원을 받았다고 밝힌다.
- Chelsea Finn과 Christopher D. Manning은 CIFAR Fellows로 표기된다.
- 본 연구는 Stanford Accelerator for Learning(SAL)과 Stanford Institute for Human-Centered Artificial Intelligence(HAI)의 *Generative AI for the Future of Learning* seed grant program의 일부 지원을 받았다.
- Stanford Center for Research on Foundation Models(CRFM)가 실험에 사용된 compute resource의 일부를 제공했다.
- ONR grant N00014-20-1-2675의 지원도 함께 명시된다.

---

## References

1. Y. Bai, A. Jones, K. Ndousse, A. Askell, A. Chen, N. DasSarma, D. Drain, S. Fort, D. Ganguli, T. Henighan, N. Joseph, S. Kadavath, J. Kernion, T. Conerly, S. El-Showk, N. Elhage, Z. HatfieldDodds, D. Hernandez, T. Hume, S. Johnston, S. Kravec, L. Lovitt, N. Nanda, C. Olsson, D. Amodei, T. Brown, J. Clark, S. McCandlish, C. Olah, B. Mann, and J. Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback, 2022.
2. Y. Bai, S. Kadavath, S. Kundu, A. Askell, J. Kernion, A. Jones, A. Chen, A. Goldie, A. Mirhoseini, C. McKinnon, C. Chen, C. Olsson, C. Olah, D. Hernandez, D. Drain, D. Ganguli, D. Li, E. Tran-Johnson, E. Perez, J. Kerr, J. Mueller, J. Ladish, J. Landau, K. Ndousse, K. Lukosuite, L. Lovitt, M. Sellitto, N. Elhage, N. Schiefer, N. Mercado, N. DasSarma, R. Lasenby, R. Larson, S. Ringer, S. Johnston, S. Kravec, S. E. Showk, S. Fort, T. Lanham, T. Telleen-Lawton, T. Conerly, T. Henighan, T. Hume, S. R. Bowman, Z. Hatfield-Dodds, B. Mann, D. Amodei, N. Joseph, S. McCandlish, T. Brown, and J. Kaplan. Constitutional ai: Harmlessness from ai feedback, 2022.
3. S. Biderman, H. Schoelkopf, Q. Anthony, H. Bradley, K. O’Brien, E. Hallahan, M. A. Khan, S. Purohit, U. S. Prashanth, E. Raff, A. Skowron, L. Sutawika, and O. van der Wal. Pythia: A suite for analyzing large language models across training and scaling, 2023.
4. H. Bong and A. Rinaldo. Generalized results for the existence and consistency of the MLE in the Bradley-Terry-Luce model. International Conference on Machine Learning, 2022. arXiv:2110.11487.
5. R. A. Bradley and M. E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952. doi: https://doi.org/10.2307/2334029.
6. T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 1877– 1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.
7. T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
8. S. Bubeck, V. Chandrasekaran, R. Eldan, J. Gehrke, E. Horvitz, E. Kamar, P. Lee, Y. T. Lee, Y. Li, S. Lundberg, H. Nori, H. Palangi, M. T. Ribeiro, and Y. Zhang. Sparks of artificial general intelligence: Early experiments with GPT-4, 2023. arXiv preprint arXiv:2303.12712.
9. R. Busa-Fekete, B. Szörényi, P. Weng, W. Cheng, and E. Hüllermeier. Preference-based reinforcement learning: evolutionary direct policy search using a preference-based racing algorithm. Machine Learning, 97(3):327–351, July 2014. doi: 10.1007/s10994-014-5458-8. URL https://doi.org/10.1007/s10994-014-5458-8.
10. Y. Chen, R. Wang, H. Jiang, S. Shi, and R.-L. Xu. Exploring the use of large language models for reference-free text quality evaluation: A preliminary empirical study. ArXiv, abs/2304.00723, 2023.
11. A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.
12. P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei. Deep reinforcement learning from human preferences. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/ paper_files/paper/2017/file/d5e2c0adad503c91f91df240d0cd4e49-Paper.pdf.
13. H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, Y. Li, X. Wang, M. Dehghani, S. Brahma, A. Webson, S. S. Gu, Z. Dai, M. Suzgun, X. Chen, A. Chowdhery, A. Castro-Ros, M. Pellat, K. Robinson, D. Valter, S. Narang, G. Mishra, A. Yu, V. Zhao, Y. Huang, A. Dai, H. Yu, S. Petrov, E. H. Chi, J. Dean, J. Devlin, A. Roberts, D. Zhou, Q. V. Le, and J. Wei. Scaling instruction-finetuned language models, 2022.
14. M. Dudík, K. Hofmann, R. E. Schapire, A. Slivkins, and M. Zoghi. Contextual dueling bandits. In P. Grünwald, E. Hazan, and S. Kale, editors, Proceedings of The 28th Conference on Learning Theory, volume 40 of Proceedings of Machine Learning Research, pages 563–587, Paris, France, 03–06 Jul 2015. PMLR. URL https://proceedings.mlr.press/v40/Dudik15.html.
15. D. Go, T. Korbak, G. Kruszewski, J. Rozen, N. Ryu, and M. Dymetman. Aligning language models with preferences through f-divergence minimization. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023.
16. A. Jain, B. Wojcik, T. Joachims, and A. Saxena. Learning trajectory preferences for manipulators via iterative improvement. In C. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Weinberger, editors, Advances in Neural Information Processing Systems, volume 26. Curran Associates, Inc., 2013. URL https://proceedings.neurips.cc/paper_files/paper/ 2013/file/c058f544c737782deacefa532d9add4c-Paper.pdf.
17. N. Jaques, S. Gu, D. Bahdanau, J. M. Hernández-Lobato, R. E. Turner, and D. Eck. Sequence tutor: Conservative fine-tuning of sequence generation models with kl-control. In International Conference on Machine Learning, pages 1645–1654. PMLR, 2017.
18. N. Jaques, J. H. Shen, A. Ghandeharioun, C. Ferguson, A. Lapedriza, N. Jones, S. S. Gu, and R. Picard. Human-centric dialog training via offline reinforcement learning. arXiv preprint arXiv:2010.05848, 2020.
19. T. Korbak, H. Elsahar, G. Kruszewski, and M. Dymetman. On reinforcement learning and distribution matching for fine-tuning language models with no catastrophic forgetting. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 16203–16220. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/file/ 67496dfa96afddab795530cc7c69b57a-Paper-Conference.pdf.
20. J. Kreutzer, J. Uyheng, and S. Riezler. Reliability and learnability of human bandit feedback for sequence-to-sequence reinforcement learning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1777–1788, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/ P18-1165. URL https://aclanthology.org/P18-1165.
21. A. Kupcsik, D. Hsu, and W. S. Lee. Learning Dynamic Robot-to-Human Object Handover from Human Feedback, pages 161–176. Springer International Publishing, 01 2018. ISBN 978-3-319-51531-1. doi: 10.1007/978-3-319-51532-8_10.
22. S. Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review, 2018.
23. R. D. Luce. Individual choice behavior: A theoretical analysis. Courier Corporation, 2012.
24. A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng, and C. Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 142–150, Portland, Oregon, USA, June 2011. Association for Computational Linguistics. URL http://www.aclweb.org/ anthology/P11-1015.
25. S. Mishra, D. Khashabi, C. Baral, and H. Hajishirzi. Cross-task generalization via natural language crowdsourcing instructions. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3470–3487, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long. 244. URL https://aclanthology.org/2022.acl-long.244.
26. R. Nallapati, B. Zhou, C. dos Santos, Ç. Gulçehre, and B. Xiang. Abstractive text summarization using sequence-to-sequence RNNs and beyond. In Proceedings of the 20th SIGNLL Conference on Computational Natural Language Learning, pages 280–290, Berlin, Germany, Aug. 2016. Association for Computational Linguistics. doi: 10.18653/v1/K16-1028. URL https:// aclanthology.org/K16-1028.
27. D. Narayanan, M. Shoeybi, J. Casper, P. LeGresley, M. Patwary, V. Korthikanti, D. Vainbrand, P. Kashinkunti, J. Bernauer, B. Catanzaro, A. Phanishayee, and M. Zaharia. Efficient large-scale language model training on gpu clusters using megatron-lm. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, SC ’21, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450384421. doi: 10.1145/3458817.3476209. URL https://doi.org/10.1145/3458817.3476209.
28. L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. F. Christiano, J. Leike, and R. Lowe. Training language models to follow instructions with human feedback. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/ paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf.
29. R. Paulus, C. Xiong, and R. Socher. A deep reinforced model for abstractive summarization. In International Conference on Learning Representations, 2018. URL https://openreview. net/forum?id=HkAClQgA-.
30. X. B. Peng, A. Kumar, G. Zhang, and S. Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.
31. J. Peters and S. Schaal. Reinforcement learning by reward-weighted regression for operational space control. In Proceedings of the 24th international conference on Machine learning, pages 745–750, 2007.
32. R. L. Plackett. The analysis of permutations. Journal of the Royal Statistical Society. Series C (Applied Statistics), 24(2):193–202, 1975. doi: https://doi.org/10.2307/2346567.
33. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. Language models are unsupervised multitask learners, 2019. Ms., OpenAI.
34. R. Ramamurthy, P. Ammanabrolu, K. Brantley, J. Hessel, R. Sifa, C. Bauckhage, H. Hajishirzi, and Y. Choi. Is reinforcement learning (not) for natural language processing: Benchmarks, baselines, and building blocks for natural language policy optimization. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview. net/forum?id=8aHzds2uUyB.
35. M. Ranzato, S. Chopra, M. Auli, and W. Zaremba. Sequence level training with recurrent neural networks. CoRR, abs/1511.06732, 2015.
36. D. Sadigh, A. D. Dragan, S. Sastry, and S. A. Seshia. Active preference-based learning of reward functions. In Robotics: Science and Systems (RSS), 2017.
37. A. Saha, A. Pacchiano, and J. Lee. Dueling rl: Reinforcement learning with trajectory preferences. In F. Ruiz, J. Dy, and J.-W. van de Meent, editors, Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 6263–6289. PMLR, 25–27 Apr 2023. URL https://proceedings.mlr.press/v206/saha23a.html.
38. V. Sanh, A. Webson, C. Raffel, S. Bach, L. Sutawika, Z. Alyafeai, A. Chaffin, A. Stiegler, A. Raja, M. Dey, M. S. Bari, C. Xu, U. Thakker, S. S. Sharma, E. Szczechla, T. Kim, G. Chhablani, N. Nayak, D. Datta, J. Chang, M. T.-J. Jiang, H. Wang, M. Manica, S. Shen, Z. X. Yong, H. Pandey, R. Bawden, T. Wang, T. Neeraj, J. Rozen, A. Sharma, A. Santilli, T. Fevry, J. A. Fries, R. Teehan, T. L. Scao, S. Biderman, L. Gao, T. Wolf, and A. M. Rush. Multitask prompted training enables zero-shot task generalization. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=9Vrb9D0WI4.
39. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms, 2017.
40. N. Stiennon, L. Ouyang, J. Wu, D. M. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, and P. Christiano. Learning to summarize from human feedback, 2022.
41. R. Thoppilan, D. D. Freitas, J. Hall, N. Shazeer, A. Kulshreshtha, H.-T. Cheng, A. Jin, T. Bos, L. Baker, Y. Du, Y. Li, H. Lee, H. S. Zheng, A. Ghafouri, M. Menegali, Y. Huang, M. Krikun, D. Lepikhin, J. Qin, D. Chen, Y. Xu, Z. Chen, A. Roberts, M. Bosma, V. Zhao, Y. Zhou, C.-C. Chang, I. Krivokon, W. Rusch, M. Pickett, P. Srinivasan, L. Man, K. Meier-Hellstern, M. R. Morris, T. Doshi, R. D. Santos, T. Duke, J. Soraker, B. Zevenbergen, V. Prabhakaran, M. Diaz, B. Hutchinson, K. Olson, A. Molina, E. Hoffman-John, J. Lee, L. Aroyo, R. Rajakumar, A. Butryna, M. Lamm, V. Kuzmina, J. Fenton, A. Cohen, R. Bernstein, R. Kurzweil, B. AgueraArcas, C. Cui, M. Croak, E. Chi, and Q. Le. Lamda: Language models for dialog applications, 2022.
42. H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
43. M. Völske, M. Potthast, S. Syed, and B. Stein. TL;DR: Mining Reddit to learn automatic summarization. In Proceedings of the Workshop on New Frontiers in Summarization, pages 59–63, Copenhagen, Denmark, Sept. 2017. Association for Computational Linguistics. doi: 10.18653/v1/W17-4508. URL https://aclanthology.org/W17-4508.
44. L. von Werra, J. Tow, reciprocated, S. Matiana, A. Havrilla, cat state, L. Castricato, Alan, D. V. Phung, A. Thakur, A. Bukhtiyarov, aaronrmm, F. Milo, Daniel, D. King, D. Shin, E. Kim, J. Wei, M. Romero, N. Pochinkov, O. Sanseviero, R. Adithyan, S. Siu, T. Simonini, V. Blagojevic, X. Song, Z. Witten, alexandremuzio, and crumb. CarperAI/trlx: v0.6.0: LLaMa (Alpaca), Benchmark Util, T5 ILQL, Tests, Mar. 2023. URL https://doi.org/10.5281/zenodo. 7790115.
45. B. Wang and A. Komatsuzaki. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/mesh-transformer-jax, May 2021.
46. S. Welleck, I. Kulikov, S. Roller, E. Dinan, K. Cho, and J. Weston. Neural text generation with unlikelihood training. arXiv preprint arXiv:1908.04319, 2019.
47. R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Mach. Learn., 8(3–4):229–256, may 1992. ISSN 0885-6125. doi: 10.1007/BF00992696. URL https://doi.org/10.1007/BF00992696.
48. Y. Wu and B. Hu. Learning to extract coherent summary via deep reinforcement learning. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence and Thirtieth Innovative Applications of Artificial Intelligence Conference and Eighth AAAI Symposium on Educational Advances in Artificial Intelligence, AAAI’18/IAAI’18/EAAI’18. AAAI Press, 2018. ISBN 978-1-57735-800-8.
49. X. Yan, C. Luo, C. L. A. Clarke, N. Craswell, E. M. Voorhees, and P. Castells. Human preferences as dueling bandits. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’22, page 567–577, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450387323. doi: 10.1145/3477495.3531991. URL https://doi.org/10.1145/3477495.3531991.
50. Y. Yue, J. Broder, R. Kleinberg, and T. Joachims. The k-armed dueling bandits problem. Journal of Computer and System Sciences, 78(5):1538–1556, 2012. ISSN 0022-0000. doi: https://doi.org/10.1016/j.jcss.2011.12.028. URL https://www.sciencedirect.com/science/ article/pii/S0022000012000281. JCSS Special Issue: Cloud Computing 2011.
51. D. M. Ziegler, N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving. Fine-tuning language models from human preferences, 2020.

---

## Author Contributions

원문은 저자 기여를 비교적 상세하게 분해하여 적는다.

- **공통 기여**: 모든 저자가 실험 설계, 분석, 반복 개선, 원고 작성 및 편집, 프로젝트 진행 관리 전반에 기여했다고 명시한다.
- **RR(Rafael Rafailov)**: EM과의 논의에서 autoregressive reward model 사용을 제안했고, DPO 목적을 유도했으며, 알고리즘의 이론적 성질을 증명하고 해당 본문·부록을 집필했다. PPO 및 reward learning baseline 일부 구성에도 기여했다.
- **AS(Archit Sharma)**: PPO 대안으로 weighted regression 계열을 사용하는 논의를 시작했고, DPO와 unlikelihood의 연결에 대한 초기 분석을 작성했다. DPO 및 baseline 구현 반복, 탐색 실험, 데이터셋·베이스라인·평가 설계, 감성 제어 및 요약 모델 학습과 평가, GPT-4 요약 평가 설계, 초록·예비지식·방법·실험 절 작성과 여타 섹션 편집에 큰 역할을 했다.
- **EM(Eric Mitchell)**: 초기 autoregressive reward model 논의에 참여했고, 최초의 DPO 구현과 초기 실험을 수행했다. 요약·대화의 대규모 DPO 모델 학습, GPT-4 win rate 인프라 구축, 인간 연구 수행과 분석, 초록·서론·관련연구·토론·실험 대부분 집필 및 나머지 섹션 편집을 담당했다.
- **CF(Chelsea Finn), CM(Christopher D. Manning), SE(Stefano Ermon)**: 연구 전체를 지도했고, 아이디어와 실험 방향을 제안했으며, 논문 집필을 도왔다.

---

# Appendix A. Mathematical Derivations

## A.1 Deriving the Optimum of the KL-Constrained Reward Maximization Objective

이 부록은 본문 Eq. (4)를 유도한다. 출발점은 다음의 일반적 최적화 문제다.

$$
\max_{\pi}
\;\mathbb{E}_{x\sim D,\;y\sim\pi}\left[r(x,y)-\beta D_{\mathrm{KL}}\bigl(\pi(y\mid x)\|\pi_{\mathrm{ref}}(y\mid x)\bigr)\right]
\tag{11}
$$

여기서 policy class는 일반적인 비모수(non-parametric) 정책 공간으로 둔다. 유도는 KL 항을 expectation 안으로 전개하는 데서 시작한다.

$$
\max_{\pi}
\;\mathbb{E}_{x\sim D}\,\mathbb{E}_{y\sim\pi(y\mid x)}
\left[r(x,y)-\beta\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}\right]
$$

이를 부호를 바꾸어 최소화 문제로 쓰면,

$$
\min_{\pi}
\;\mathbb{E}_{x\sim D}\,\mathbb{E}_{y\sim\pi(y\mid x)}
\left[\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}-\frac{1}{\beta}r(x,y)\right]
$$

이고, 여기에 partition function

$$
Z(x)=\sum_y \pi_{\mathrm{ref}}(y\mid x)\exp\!\left(\frac{1}{\beta}r(x,y)\right)
$$

를 도입하면 다음과 같이 다시 쓸 수 있다.

$$
\min_{\pi}
\;\mathbb{E}_{x\sim D}\,\mathbb{E}_{y\sim\pi(y\mid x)}
\left[
\log\frac{\pi(y\mid x)}{\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\exp(\frac{1}{\beta}r(x,y))}
-\log Z(x)
\right]
\tag{12}
$$

이제

$$
\pi^*(y\mid x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\exp\!\left(\frac{1}{\beta}r(x,y)\right)
$$

를 정의하면 이는 유효한 확률분포가 된다. 따라서 Eq. (12)는

$$
\min_{\pi}
\;\mathbb{E}_{x\sim D}\left[D_{\mathrm{KL}}\bigl(\pi(y\mid x)\|\pi^*(y\mid x)\bigr)-\log Z(x)\right]
\tag{14}
$$

로 정리된다. \(Z(x)\)는 \(\pi\)와 무관하므로 최소값은 KL 항을 0으로 만드는 경우, 즉 \(\pi=\pi^*\)에서 달성된다. 따라서 최적 정책은

$$
\pi(y\mid x)=\pi^*(y\mid x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\exp\!\left(\frac{1}{\beta}r(x,y)\right)
\tag{15}
$$

가 된다. 이는 본문 Eq. (4)의 재진술이다.

## A.2 Deriving the DPO Objective Under the Bradley–Terry Model

부록 A.2는 Bradley–Terry 모델에서 Eq. (7)이 어떻게 나오는지 직접 계산한다. 시작점은

$$
p^*(y_1\succ y_2\mid x)=\frac{\exp(r^*(x,y_1))}{\exp(r^*(x,y_1)) + \exp(r^*(x,y_2))}
\tag{16}
$$

이며, 본문에서 이미 얻은

$$
r^*(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} + \beta \log Z(x)
\tag{17}
$$

를 대입한다. 그러면 분자와 분모에 공통으로 포함된 \(\beta\log Z(x)\)가 상쇄되고,

$$
p^*(y_1\succ y_2\mid x)
=
\sigma\!\left(
\beta\log\frac{\pi^*(y_1\mid x)}{\pi_{\mathrm{ref}}(y_1\mid x)}
-
\beta\log\frac{\pi^*(y_2\mid x)}{\pi_{\mathrm{ref}}(y_2\mid x)}
\right)
$$

형태가 남는다. 이 식에 winner와 loser를 넣고 최대우도 학습을 취하면 본문 Eq. (7)의 per-instance loss가 얻어진다. 부록 A.2는 결국 본문 derivation의 핵심인 “partition function cancellation”을 명시적으로 보이는 절이다.

## A.3 Deriving the DPO Objective Under the Plackett–Luce Model

논문은 pairwise 선호를 넘어, \(K\)개의 응답에 대한 순위 전체를 다루는 Plackett–Luce 모델도 같은 방식으로 확장 가능하다고 설명한다. 프롬프트 \(x\)와 응답 집합 \(y_1,\dots,y_K\)가 주어졌을 때, 사용자의 순위 \(\tau:[K]\to[K]\)에 대한 확률은

$$
p^*(\tau\mid y_1,\dots,y_K,x)=
\prod_{k=1}^{K}
\frac{\exp(r^*(x,y_{\tau(k)}))}{\sum_{j=k}^{K}\exp(r^*(x,y_{\tau(j)}))}
\tag{18}
$$

로 정의된다. \(K=2\)이면 Bradley–Terry가 된다. 여기에도 Eq. (5)와 같은 optimal-policy parameterization을 대입하면, 역시 \(Z(x)\)가 소거되고 다음 식이 남는다.

$$
p^*(\tau\mid y_1,\dots,y_K,x)=
\prod_{k=1}^{K}
\frac{\exp\left(\beta\log\frac{\pi^*(y_{\tau(k)}\mid x)}{\pi_{\mathrm{ref}}(y_{\tau(k)}\mid x)}\right)}
{\sum_{j=k}^{K}\exp\left(\beta\log\frac{\pi^*(y_{\tau(j)}\mid x)}{\pi_{\mathrm{ref}}(y_{\tau(j)}\mid x)}\right)}
\tag{19}
$$

이제 랭킹 데이터셋

$$
D=\{(\tau^{(i)}, y_1^{(i)},\dots,y_K^{(i)}, x^{(i)})\}_{i=1}^{N}
$$

이 있다면, 파라미터화된 정책 \(\pi_\theta\)에 대해 다음 최대우도 목적을 쓸 수 있다.

$$
L_{\mathrm{DPO}}(\pi_\theta,\pi_{\mathrm{ref}})
=
-\mathbb{E}_{\tau,y_1,\dots,y_K,x\sim D}
\left[
\log
\prod_{k=1}^{K}
\frac{\exp\left(\beta\log\frac{\pi_\theta(y_{\tau(k)}\mid x)}{\pi_{\mathrm{ref}}(y_{\tau(k)}\mid x)}\right)}
{\sum_{j=k}^{K}\exp\left(\beta\log\frac{\pi_\theta(y_{\tau(j)}\mid x)}{\pi_{\mathrm{ref}}(y_{\tau(j)}\mid x)}\right)}
\right]
\tag{20}
$$

즉, DPO의 핵심은 Bradley–Terry에 국한된 발상이 아니라, 보다 일반적인 랭킹 선호모델에도 자연스럽게 들어맞는다.

## A.4 Deriving the Gradient of the DPO Objective

부록 A.4는 Section 4의 gradient 식을 단계적으로 유도한다. 먼저 DPO 목적을 다시 적으면,

$$
\nabla_\theta L_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}})
=
-\nabla_\theta
\mathbb{E}_{(x,y_w,y_l)\sim D}
\left[
\log\sigma\left(
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right)
\right]
\tag{21}
$$

이다. 저자들은 내부 스칼라를

$$
u =
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
$$

로 두고,

$$
\nabla_\theta L_{\mathrm{DPO}}
=
-\mathbb{E}_{(x,y_w,y_l)\sim D}
\left[
\frac{\sigma'(\nu)}{\sigma(\nu)}\nabla_\theta \nu
\right]
\tag{22}
$$

로 바꾼다. 이후 sigmoid의 성질 \(\sigma'(x)=\sigma(x)(1-\sigma(x))\)와 \(\sigma(-x)=1-\sigma(x)\)를 사용하면, 본문에서 제시한 gradient 형태가 나온다. 이 과정은 동적 가중치가 로지스틱 도함수에서 직접 유도된다는 사실을 분명히 보여 준다.

## A.5 Proof of Lemma 1 and 2

### Lemma 1의 증명

가정은 \(r'(x,y)=r(x,y)+f(x)\)이다. Plackett–Luce 분포를 \(p_r\)라 하면,

$$
p_{r'}(\tau\mid y_1,\dots,y_K,x)
=
\prod_{k=1}^{K}
\frac{\exp(r'(x,y_{\tau(k)}))}{\sum_{j=k}^{K}\exp(r'(x,y_{\tau(j)}))}
$$

이고, 여기에 \(r'=r+f(x)\)를 대입하면 각 분자와 분모에 \(\exp(f(x))\)가 공통으로 곱해지므로 완전히 소거된다. 결과적으로 \(p_{r'}=p_r\)이며, 같은 class의 reward는 동일한 preference distribution을 유도한다.

### Lemma 2의 증명

동일한 가정 아래, Eq. (4)의 optimal policy 공식을 \(r'\)에 적용하면

$$
\pi_{r'}(y\mid x)
=
\frac{\pi_{\mathrm{ref}}(y\mid x)\exp\!\left(\frac{1}{\beta}(r(x,y)+f(x))\right)}
{\sum_y \pi_{\mathrm{ref}}(y\mid x)\exp\!\left(\frac{1}{\beta}(r(x,y)+f(x))\right)}
$$

인데, 분자와 분모에 공통인 \(\exp(f(x)/\beta)\)가 again 상쇄되어

$$
\pi_{r'}(y\mid x)=\pi_r(y\mid x)
$$

가 된다. 따라서 동일한 reward class는 동일한 optimal policy를 유도한다.

## A.6 Proof of Theorem 1

부록 A.6은 Theorem 1을 보다 엄밀하게 전개한다. 가정은 다음 두 가지다.

1. reference model \(\pi_{\mathrm{ref}}(y\mid x)>0\)가 모든 prompt–answer 쌍에 대해 성립한다.
2. \(\beta>0\)이다.

임의의 reward \(r(x,y)\)를 잡고, Eq. (4)에 의해 이 reward가 유도하는 optimal policy를 \(\pi_r(y\mid x)\)라 하자. Eq. (5)에 따르면

$$
r(x,y)=\beta\log\frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} + \beta\log Z(x)
$$

이므로,

$$
r'(x,y)=f(r,\pi_{\mathrm{ref}},\beta)(x,y)=r(x,y)-\beta\log Z(x)
$$

를 정의하면 \(r'\)는 \(r\)와 같은 equivalence class에 있으면서

$$
r'(x,y)=\beta\log\frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
$$

꼴을 갖는다. 즉, 어떤 reward class든 DPO 재파라미터화로 대표자를 잡을 수 있다.

부록은 여기서 한 걸음 더 나아가 **Proposition 1**을 제시한다. 모든 equivalence class마다 위 형태로 표현되는 reward는 유일하다. 증명은 귀류법으로 주어진다. 만약 같은 class 안에 두 개의 서로 다른 reward가 각각 서로 다른 정책 \(\pi\)와 \(\pi'\)에 대해

$$
r(x,y)=\beta\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)},
\qquad
r'(x,y)=\beta\log\frac{\pi'(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
$$

로 표현된다고 하자. 그런데 같은 class라는 것은 \(r'=r+f(x)\)이므로, 두 식을 합치면

$$
\pi'(y\mid x)=\pi(y\mid x)\exp\left(\frac{1}{\beta}f(x)\right)
$$

를 얻는다. 양변을 \(y\)에 대해 합하면 두 분포의 합이 모두 1이어야 하므로 \(\exp(f(x)/\beta)=1\), 곧 \(f(x)=0\)이 되어 \(r=r'\)가 된다. 따라서 대표자는 유일하다.

---

# Appendix B. DPO Implementation Details and Hyperparameters

원문은 DPO가 구현상으로도 단순하다는 점을 강조하기 위해, PyTorch 코드 스니펫 수준의 손실 계산 예시를 제시한다. 그 핵심 연산을 그대로 옮기면 다음과 같다.

### DPO 손실 계산 절차(원문 코드의 의미 재구성)

**입력**
- `pi_logps`: 현재 정책이 각 completion에 부여한 log probability
- `ref_logps`: reference model이 각 completion에 부여한 log probability
- `yw_idxs`: winner completion의 인덱스들
- `yl_idxs`: loser completion의 인덱스들
- `beta`: KL 제약 강도를 조절하는 계수

**절차**
1. winner와 loser에 해당하는 정책 log-probability를 각각 추출한다.
2. 같은 방식으로 reference model의 winner/loser log-probability를 추출한다.
3. 정책의 log-ratio
   $$
   \log\pi_\theta(y_w\mid x)-\log\pi_\theta(y_l\mid x)
   $$
   와 reference의 log-ratio
   $$
   \log\pi_{\mathrm{ref}}(y_w\mid x)-\log\pi_{\mathrm{ref}}(y_l\mid x)
   $$
   를 계산한다.
4. 두 log-ratio의 차이에 \(\beta\)를 곱한 뒤,
   $$
   -\log \sigma\bigl(\beta[(\log\pi_w-\log\pi_l)-(\log\pi^{\mathrm{ref}}_w-\log\pi^{\mathrm{ref}}_l)]\bigr)
   $$
   를 per-pair loss로 사용한다.
5. 보상처럼 기록하는 값은
   $$
   \beta\,(\log\pi_\theta(y\mid x)-\log\pi_{\mathrm{ref}}(y\mid x))
   $$
   으로 계산하되, gradient는 흘리지 않도록 detach한다.

원문이 제시한 기본 하이퍼파라미터는 다음과 같다.

| 항목 | 기본값 |
|---|---:|
| \(\beta\) | 0.1 |
| 배치 크기 | 64 |
| 최적화기 | RMSprop |
| 학습률 | \(1\times10^{-6}\) |
| warmup | 150 step 동안 0에서 \(1\times10^{-6}\)까지 선형 증가 |
| TL;DR 요약에서의 \(\beta\) | 0.5 |

즉, 요약 과제를 제외하면 기본 설정은 \(\beta=0.1\), batch size 64, RMSprop, learning rate \(10^{-6}\), 150-step linear warmup이다.

---

# Appendix C. Further Details on the Experimental Set-Up

## C.1 IMDb Sentiment Experiment and Baseline Details

원문은 감성 제어 실험의 상세를 다음과 같이 기록한다.

- 프롬프트는 IMDB 데이터셋에서 길이 **2–8 토큰**인 prefix로 구성한다.
- ground-truth reward model로는 `siebert/sentiment-roberta-large-english` 사전학습 감성 분류기를 사용한다.
- base language model은 `gpt2-large`다.
- 저자들은 기본적으로 많이 쓰이는 더 작은 설정이 저품질 텍스트를 만들고 reward도 다소 부정확하다고 판단하여, 더 큰 모델을 사용했다고 설명한다.
- 먼저 IMDB 데이터의 일부에 대해 **1 epoch** supervised fine-tuning을 수행한다.
- 이후 이 모델로 **25,000개 prefix** 각각에 대해 **4개 completion**을 샘플링한다.
- 각 prefix마다 ground-truth reward model을 이용해 **6개의 preference pair**를 만든다.
- RLHF reward model은 `gpt2-large`에서 초기화하고, preference dataset 위에서 **3 epoch** 학습한다.
- reward model 체크포인트는 **validation accuracy가 가장 높은 지점**을 선택한다.
- “TRL” run은 TRL 라이브러리의 기본 hyperparameter를 그대로 사용한다.
- 저자들의 PPO 구현은 PPO step당 **1024개의 대형 배치 샘플**을 사용한다.

이 부록은 본문 Figure 2 왼쪽 결과가 어떤 데이터 구성과 어떤 보상 정의 위에 서 있는지 구체적으로 밝혀 준다.

## C.2 GPT-4 prompts for computing summarization and dialogue win rates

원문은 GPT-4 자동 평가에 사용한 프롬프트를 전문으로 제시한다. 모델은 모든 실험에서 `gpt-4-0314`이며, 평가 시 summary 또는 response의 A/B 순서는 매번 무작위로 바뀐다.

### 요약 평가 프롬프트 (S)

단순 프롬프트는 다음 구조를 갖는다.

1. 질문: 주어진 포럼 글의 가장 중요한 요점을 더 잘 요약한 summary가 어느 것인가?
2. 입력 슬롯: `Post`, `Summary A`, `Summary B`
3. 출력 형식:
   - 첫 줄: 두 요약을 한 문장으로 비교하고, 어떤 쪽을 선호하는지와 그 이유를 설명한다.
   - 둘째 줄: `Preferred: A` 혹은 `Preferred: B`만 출력한다.

### 요약 평가 프롬프트 (C)

간결성 프롬프트는 위 구조에 다음 문장을 추가한다.

- 좋은 요약은 가장 중요한 요점을 담되, **불필요하거나 무관한 세부를 포함하지 않아야 하며**, 정확하면서도 간결해야 한다.

출력 형식은 (S)와 동일하다. 본문이 말하듯, 이 프롬프트는 GPT-4가 길고 반복적인 요약을 선호하는 경향을 보정하기 위해 도입되었다.

### 대화 평가 프롬프트

대화 프롬프트는 다음과 같다.

1. 질문: 주어진 챗봇 질의에 대해 어느 응답이 더 helpful한가?
2. 입력 슬롯: `Query`, `Response A`, `Response B`
3. 출력 형식:
   - 첫 줄: 두 응답을 한 문장으로 비교하고, 어느 쪽이 더 helpful한지 이유를 덧붙인다.
   - 둘째 줄: `More helpful: A` 혹은 `More helpful: B`만 출력한다.

평가 프로토콜에서 핵심은, 모델이 먼저 **비교 이유를 한 문장으로 기술한 다음** 마지막 줄에 선택만 남기도록 강제했다는 점이다.

## C.3 Unlikelihood baseline

원문은 감성 실험에서는 unlikelihood baseline을 포함하지만, 요약과 대화에서는 제외한다고 밝힌다. 이 baseline은 단순히

- winner response의 \(\log p(y_w\mid x)\)는 최대화하고,
- loser response의 \(\log p(y_l\mid x)\)는 최소화하는

형태다. 저자들은 이 접근이 보다 복잡한 생성 과제에서는 **의미 없는 응답을 생성**하는 경향이 강하다고 보고, summarization과 dialogue 실험에서는 baseline으로 채택하지 않는다.

### Table 3 해설

> Table 3 삽입

Table 3은 TL;DR 프롬프트에 대해 temperature 1.0으로 샘플했을 때 unlikelihood가 실제로 어떤 출력을 내는지 보여 준다. 표에는 두 개의 Reddit 예시가 들어 있다.

1. **r/relationships 예시**: 한 달 정도 만난 상대가 하루 종일 답장을 하지 않았고, 친구와 시간을 보낸 정황을 본 뒤 불안해하는 내용이다. 그러나 unlikelihood 출력은 `girl when when when ...` 식의 반복 문자열로 무너진다.
2. **r/tifu 예시**: 장례식장에서 실수로 노인을 걷어찬 사건을 회상하는 글이다. 여기서도 출력은 `when an old woman was tripping ... when when when ...`처럼 반복 토큰으로 붕괴한다.

표의 캡션은 이 현상을 “more complex problems such as summarization and dialogue”에서 meaningful response를 생성하지 못하는 사례라고 규정한다. 본문 Section 4가 언급한 “나이브 목적의 degeneration”이 실제 텍스트 생성에서는 어떤 모양으로 나타나는지를 보여 주는 부록 자료라고 볼 수 있다.

# Appendix D. Additional Empirical Results

## D.1 Performance of Best of N baseline for Various N

부록 D.1은 Best of \(N\) baseline의 성능이 \(N\)에 따라 어떻게 바뀌는지를 보여 준다. 논문은 이 baseline이 강력하지만 계산량이 크다는 점을 다시 강조하면서, Anthropic-HH 대화와 TL;DR 요약 모두에서 여러 \(N\)을 비교한다.

### Figure 4 해설

> Figure 4 삽입

Figure 4는 두 패널로 구성된다.

- **왼쪽(Anthropic-HH Dialogue Win Rate vs Chosen)**: \(N=\{1,4,16,64,128\}\)를 비교한다. temperature가 0.25, 0.75, 1.0으로 변할 때 win rate가 상승하지만, 성능 개선은 대체로 \(N=64\) 혹은 \(128\) 부근에서 포화된다.
- **오른쪽(TL;DR Summarization Win Rate vs Reference)**: \(N=64,128,256\)을 비교한다. 0.25–0.75 구간에서 높은 성능을 보이지만, \(N\)을 128에서 256으로 늘려도 개선 폭은 매우 제한적이다.

캡션은 이를 “roughly 64–128 samples” 수준에서 plateau한다고 요약한다. 즉, Best of \(N\)은 충분한 샘플 수를 주면 강하지만, 추가 샘플의 한계효용이 급격히 줄어드는 계산량 대비 성능 구조를 가진다.

## D.2 Sample Responses and GPT-4 Judgments

이 절은 요약에서는 DPO와 PPO(temp 0), 대화에서는 DPO와 데이터셋의 chosen completion(GT)을 비교한 사례를 제시한다. 저자들은 GPT-4 판단이 어떤 종류의 품질 차이를 포착하는지, 또 어디서 실수하는지를 정성적으로 보여 주려 한다.

### Table 4 — 요약 예시 1

> Table 4 삽입

- **원문 프롬프트**: r/AskReddit 글로, 곧 대학에 떠나는 17세 사용자가 그동안 부모에게 못되게 굴었던 것을 후회하며, 생일이나 크리스마스 같은 특별한 날에 무엇을 하면 부모를 기쁘게 할 수 있을지 묻는다.
- **DPO 요약**: 부모에게 상처를 준 것을 만회하고 싶으며, 특별한 날 어떤 행동으로 사랑과 감사를 표현할 수 있을지 묻는 내용으로 정리한다.
- **PPO 요약**: 대학 이후에도 무엇을 할 수 있는지 등을 반복적으로 언급하면서 문장이 중복되고 덜 선명하다.
- **GPT-4 판단**: DPO 쪽이 핵심을 더 잘 포착했고, 특별한 날 부모를 기쁘게 하는 행동이라는 중심 의제를 더 명확하게 드러낸다고 본다.

### Table 5 — 요약 예시 2

> Table 5 삽입

- **원문 프롬프트**: 2년 가까이 사귄 여자친구에게 곧 청혼할 예정인데 지나치게 긴장되어 잠을 잘 수 없고, 그 전까지 어떻게 진정해야 할지 묻는 글이다.
- **DPO 요약**: “곧 청혼할 예정인데 극도로 불안해서 잠을 이루지 못한다. 어떻게 진정할 수 있을까?”로 간결하게 압축한다.
- **PPO 요약**: 핵심은 비슷하지만 줄 바꿈과 반복적 표현이 포함되어 응답이 다소 지저분하다.
- **GPT-4 판단**: DPO가 질문자의 불안과 ‘이완 방법을 찾는다’는 목적을 더 간결하고 명확하게 담아냈다고 평가한다.

### Table 6 — 요약 예시 3

> Table 6 삽입

- **원문 프롬프트**: 저칼로리 식단을 위해 고섬유질 파스타를 고려하는 글이다. 작성자는 체중 감량보다도 “칼로리를 줄이되 파스타를 계속 먹을 수 있는가”라는 문제를 묻는다.
- **DPO 요약**: 파스타가 주요 칼로리 공급원이어서 줄이려 하는데, 고섬유 파스타가 일반 파스타와 비슷한 대체재인지 묻는 것으로 요약한다.
- **PPO 요약**: “살을 빼고 싶다”는 동기를 전면에 세워 원문에 없던 weight-loss 의도를 가정한다.
- **GPT-4 판단**: DPO가 저칼로리 대체재 탐색이라는 실제 의도를 더 정확히 포착했으며, 체중 감량이라는 해석을 함부로 덧씌우지 않았다고 본다.

### Table 7 — 대화 예시 1: GPT-4 chooses DPO over GT

> Table 7 삽입

- **질문**: “미국 시민권 운동(Civil Rights Movement)에 대한 에세이를 쓰는 데 도움을 줄 수 있나?”
- **DPO 응답**: 1950~1980년대의 시민권 운동, 법적 분리 해소, Civil Rights Act of 1964, 경제적 평등과 인종 정의 등을 길게 설명한 뒤, 어떤 유형의 에세이를 원하는지 되묻는다. 길이 제한에 걸려 잘린다.
- **GT 응답**: 직접 에세이를 써 주기보다, 먼저 다루고 싶은 핵심 포인트가 무엇인지 질문한다.
- **GPT-4 판단**: DPO 응답이 운동에 대한 구체적 정보를 제공하고 에세이 주제 선택도 도와준다는 점에서 더 helpful하다고 평가한다.

### Table 8 — 대화 예시 2: GPT-4 chooses DPO over GT

> Table 8 삽입

- **질문**: Hyatt 가문의 상속인 J.B. Pritzker의 주거지 주소를 찾고 싶다는 요청이다.
- **DPO 응답**: 동의 없는 개인 정보 추적은 불법적이고 무책임하므로 도와줄 수 없다고 거절한다.
- **GT 응답**: 주소는 주지 않지만 Hyatt 공동 설립자이자 Forbes 400 등 배경 정보를 말한다.
- **GPT-4 판단**: DPO가 사생활 보호와 법적 문제를 더 분명하게 강조한다고 본다.

### Table 9 — 대화 예시 3: GPT-4 chooses GT over DPO

> Table 9 삽입

- **질문**: “무엇이 미국을 제2차 세계대전에 참전하게 했는가?”
- **DPO 응답**: 대공황, 국제 부채, 일본의 중국 공격, 이른바 ‘coalition of the willing’, ‘all-inclusive association’ 같은 잘못된 개념을 섞어 길고 그럴듯하지만 사실과 다른 설명을 한다.
- **GT 응답**: 일본의 진주만 공격 이후 미국이 일본에 선전포고했다는 직접적이고 정확한 답을 준다.
- **GPT-4 판단**: GT가 더 직접적이고 정확하다고 평가한다.
- **캡션의 핵심**: DPO 응답은 verbose하고 plausible하지만 사실 오류를 포함한다. 저자들은 `coalition of the willing`이 WWII 맥락의 개념이 아니며 `all-inclusive association`도 실제 조직이 아니라고 명시한다.

### Table 10 — 대화 예시 4: GPT-4 chooses GT over DPO, but GPT-4 is wrong

> Table 10 삽입

- **질문**: “7 plus 2는 얼마인가?”
- **DPO 응답**: 매우 장황하게 ‘9’라고 말하고, 의미 없는 arithmetic-prefix 이야기를 이어 간다.
- **GT 응답**: `11`
- **GPT-4 판단**: GT가 direct하고 accurate하다고 판단한다.
- **캡션의 핵심**: 여기서는 GPT-4가 명백히 틀린 평가를 한다. GT의 `11`은 오답이고, DPO는 장황하지만 실제 정답 `9`를 말하고 있다. 이 표는 자동 평가자의 한계를 의도적으로 드러낸다.

## D.3 Human study details

부록 D.3는 GPT-4 평가를 인간 판단으로 교차 검증한 실험의 설계를 상세히 밝힌다.

### 실험 구성

- 비교 대상은 TL;DR 요약의 세 매치업이다.
  1. DPO (temp. 0.25) vs PPO (temp. 0.0)
  2. SFT (temp. 0.25) vs PPO (temp. 0.0)
  3. PPO (temp. 1.0) vs PPO (temp. 0.0)
- 서로 다른 품질대의 알고리즘을 고른 이유는, response quality spectrum 전반에서 인간과 GPT-4의 일치 정도를 보기 위해서다.
- 샘플 수는 다음과 같다.
  - DPO vs PPO-0: **150개 random comparison**, 각 비교에 **인간 2명** 배정 → 총 **275 judgment**
  - PPO-1 vs PPO-0: **100개 random comparison**, 각 비교에 **인간 2명** 배정 → 총 **200 judgment**
  - SFT vs PPO-0: **125개 비교**, **인간 1명** 배정
- tie로 표시된 판단은 전체의 약 **1%** 수준이며, 분석에서는 제외한다.
- 두 인간 평가자가 모두 있는 경우에는 human–human raw agreement를 계산하고, 모든 경우에 human–GPT-4 agreement도 측정한다.

### Figure 5 해설

> Figure 5 삽입

Figure 5는 SurveyMonkey에서 사용된 실제 평가 화면을 보여 준다. 각 응답자는 25개의 동일 형식 판단을 수행했으며, 질문에는 원문 포스트, Summary A, Summary B, 그리고 “거의 동일하다면 I can’t tell을 쓰라”는 지침이 포함되어 있다. 부록은 자동 평가와 인간 평가가 같은 비교 과제를 얼마나 유사하게 수행하는지 보이기 위해 이 UI를 제시한다.

### Participants

부록은 총 **25명의 자원자 인간 평가자**가 참여했다고 밝힌다. 다만 1명은 설문을 늦게 완료하여 최종 분석에는 포함되지 않았지만, 명단에는 기재되어 있다. 평가자들은 Stanford 학생(학부부터 박사 과정), 최근 Stanford 졸업생, 또는 방문 연구자로 구성되어 있으며, STEM—특히 CS—배경이 주를 이룬다.

원문이 감사와 함께 열거한 참가자 명단은 다음과 같다.

1. Gordon Chi  
2. Virginia Adams  
3. Max Du  
4. Kaili Huang  
5. Ben Prystawski  
6. Ioanna Vavelidou  
7. Victor Kolev  
8. Karel D’Oosterlinck  
9. Ananth Agarwal  
10. Tyler Lum  
11. Mike Hardy  
12. Niveditha Iyer  
13. Helena Vasconcelos  
14. Katherine Li  
15. Chenchen Gu  
16. Moritz Stephan  
17. Swee Kiat Lim  
18. Ethan Chi  
19. Kaien Yang  
20. Ryan Chi  
21. Joy Yun  
22. Abhay Singhal  
23. Siyan Li  
24. Amelia Hardy  
25. Zhengxuan Wu

원문 각주는 DPO–PPO 비교에 대해 **한 명의 자원자가 응답하지 않았다**고 덧붙인다.

---

## 맺음말

이상으로 논문의 초록, 본문, 그림·표 캡션, 감사의 글, 참고문헌 앞뒤의 보조 자료, 저자 기여, 그리고 Appendix A–D의 세부 사항까지 전부 재구성하였다. 논문의 핵심 기여는 RLHF를 다른 최적화 문제로 단순 치환한 것이 아니라, reward와 policy의 관계를 재해석하여 **선호학습 문제를 직접적인 정책 학습 문제로 다시 쓴 점**에 있다. 실험과 부록은 그 단순화가 이론적으로도 정당화되고, 실제 학습 프로토콜에서도 상당한 이점을 가질 수 있음을 보여 주도록 구성되어 있다.
