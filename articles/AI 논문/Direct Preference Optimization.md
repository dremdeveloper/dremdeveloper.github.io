# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

## Abstract

대규모 비지도 언어 모델은 폭넓은 세계 지식과 일정 수준의 추론 능력을 학습하지만, 완전히 비지도인 사전학습만으로는 원하는 행동을 정밀하게 통제하기 어렵다. 기존의 정렬 방법은 모델이 생성한 여러 응답에 대해 인간이 상대적 선호를 표시한 데이터를 모은 뒤, 먼저 그 선호를 설명하는 보상 모델을 학습하고 다시 강화학습으로 정책을 미세조정하는 RLHF 파이프라인을 사용한다. 그러나 RLHF는 보상 모델과 정책을 분리해 학습해야 하고, 학습 도중 정책 샘플링과 강화학습 안정화 기법이 계속 필요하므로 구현과 튜닝이 복잡하다.

이 논문은 RLHF의 보상 함수를 다른 방식으로 매개변수화하면 대응하는 최적 정책을 닫힌형태로 직접 쓸 수 있음을 보인다. 그 결과 표준 RLHF 문제를 별도의 강화학습 없이 단순한 분류 손실로 풀 수 있으며, 저자들은 이 방법을 **Direct Preference Optimization (DPO)** 라고 부른다. DPO는 안정적이고 계산량이 가볍고, 미세조정 중 반복 샘플링이나 대규모 하이퍼파라미터 탐색이 거의 필요 없다. 실험에서는 감성 제어, 요약, 단일 턴 대화에서 DPO가 PPO 기반 RLHF에 필적하거나 더 나은 성능을 보인다고 보고한다.

## 1. Introduction

대규모 언어 모델은 거대한 말뭉치로부터 예상보다 넓은 능력을 학습한다. 하지만 그 말뭉치는 서로 다른 목표와 우선순위, 숙련도를 가진 인간이 만든 텍스트의 혼합물이기 때문에, 모델이 무엇을 이해하고 있는가와 실제 생성 시 무엇을 더 자주 말하도록 만들어야 하는가는 다르다. 예를 들어 코딩 보조 모델은 흔한 오류를 이해해야 하지만, 생성 단계에서는 가능한 한 더 정확하고 질 높은 코드 쪽으로 편향되기를 바란다. 어떤 사회적 오개념을 모델이 "알고" 있는 것과, 그 오개념을 사실처럼 반복하는 것은 전혀 같은 일이 아니다. 결국 정렬 문제는 모델의 지식 전체를 바꾸는 문제가 아니라, 이미 품고 있는 매우 넓은 능력 공간에서 어떤 행동과 어떤 응답을 선택하도록 만들 것인가의 문제다.

[Figure 1 원문 이미지 및 원문 캡션 삽입]

기존의 선호 정렬 방법은 주로 강화학습을 사용한다. 보통 먼저 사람이 선호하는 응답과 덜 선호하는 응답을 모아 보상 모델을 만든 뒤, 그 보상을 최대화하는 방향으로 정책을 업데이트한다. 이때 정책이 원래 언어 모델에서 너무 멀리 벗어나면 보상 모델이 신뢰할 수 있는 범위를 벗어나기 때문에 KL 제약이나 유사한 규제가 반드시 따라온다. 문제는 이 전체 과정이 감독학습보다 훨씬 복잡하다는 점이다. 보상 모델과 정책이 분리되어 있고, 정책은 학습 과정에서 계속 샘플링해야 하며, PPO 같은 RL 알고리즘은 학습률, KL 타깃, 보상 정규화, value baseline 등 많은 세부 설정에 민감하다.

논문의 핵심 주장은 RLHF의 표준 목적 자체는 유지하면서도, 그 목적을 강화학습 없이 직접 최적화할 수 있다는 것이다. 저자들은 인간 선호를 설명하는 Bradley-Terry 계열 선호 모델을 그대로 받아들이되, 보상 함수를 먼저 학습한 다음 RL로 정책을 찾는 두 단계 구조를 정책 자체에 대한 직접 최적화 문제로 바꾼다. 이 변수변환이 가능한 이유는 KL 제약이 있는 보상 최대화 문제의 최적 정책이 닫힌형태로 쓰이기 때문이다. 즉, 정책이 곧 암묵적인 보상 모델 역할까지 맡을 수 있다.

논문이 강조하는 또 다른 포인트는 DPO가 단순히 "좋은 응답의 확률을 키우고 나쁜 응답의 확률을 줄이는" 순진한 방식이 아니라는 점이다. DPO는 선호된 응답과 비선호 응답의 상대 로그확률 차이를 조절하면서, 각 예제가 현재 정책에서 얼마나 잘 정렬되어 있는지에 따라 동적으로 가중을 둔다. 이 가중은 나중에 기울기를 전개했을 때 분명해지는데, 이미 올바르게 정렬된 쌍에는 작은 업데이트를, 아직 잘못된 순서를 유지하는 쌍에는 큰 업데이트를 주는 형태다. 저자들은 이러한 가중이 빠진 순진한 목적은 모델 퇴화를 일으킬 수 있다고 주장한다.

정리하면, 이 논문의 기여는 세 가지 축으로 읽을 수 있다. 첫째, RLHF의 KL 제약 보상 최대화 문제를 정책 직접 최적화 문제로 다시 쓰는 이론적 재정식화다. 둘째, 그 결과로 얻는 DPO라는 RL-free 학습 알고리즘이다. 셋째, 감성 제어, 요약, 대화 정렬 실험에서 DPO가 PPO 계열 RLHF와 비슷하거나 더 나은 성능을 내면서도 구현과 튜닝이 훨씬 간단하다는 경험적 결과다.

## 2. Related Work

저자들은 관련 연구를 네 갈래로 정리한다. 첫 번째 갈래는 자기지도 사전학습과 instruction tuning이다. 대규모 자기지도 언어 모델은 제로샷 또는 퓨샷 프롬프팅만으로도 상당한 성능을 보이지만, 사람이 작성한 지시문과 응답으로 미세조정하면 사용자 의도와 훨씬 더 잘 맞는 행동을 하게 된다. instruction tuning은 모델이 훈련 세트 밖의 지시로도 일반화할 수 있도록 만들어 실제 사용성을 크게 높여 왔다.

두 번째 갈래는 인간 선호 데이터로 언어 모델을 정렬하는 RLHF/RLAIF 계열이다. 이 방법들은 보통 두 응답 중 어느 쪽이 더 좋은지에 대한 상대 평가를 수집하고, Bradley-Terry 같은 선호 모델 아래에서 보상 모델을 학습한 뒤, PPO나 REINFORCE 계열 방법으로 언어 모델을 최적화한다. 번역, 요약, 스토리텔링, instruction following 등 다양한 과제에서 이러한 접근이 성과를 냈다. 또한 Constitution이나 rubric 같은 약한 인간 감독만으로 AI가 더 많은 선호 데이터를 생성하는 RLAIF 계열도 등장했다.

세 번째 갈래는 언어 바깥에서의 선호 기반 정책 학습이다. contextual dueling bandit은 절대 보상 대신 행동 간의 선호 관계를 사용하고, preference-based RL은 알려지지 않은 점수 함수가 만드는 이진 선호를 이용해 정책을 학습한다. 하지만 이들 역시 대체로 잠재 점수 함수, 다시 말해 보상 모델을 먼저 추정한 다음 정책을 최적화한다. 즉, 보상 추정과 정책 개선이 분리되어 있다는 점에서는 RLHF와 구조가 비슷하다.

네 번째 갈래는 제어를 추론의 한 형태로 보는 control-as-inference 관점과 KL-regularized RL이다. 이 관점에서는 보상 최대화와 기준 정책으로부터의 이탈 비용이 함께 등장하며, 최적 정책이 지수적 기울기 형태를 가진다. DPO는 바로 이 구조를 선호 학습과 연결한다. 따라서 이 논문의 새로움은 선호 모델 자체를 새로 만든 것이 아니라, 이미 알려진 선호 모델과 KL 제약 보상 최적화의 결합에서 보상 모델과 RL 루프를 제거해도 되는 매개변수화를 발견했다는 데 있다.

## 3. Preliminaries

이 절에서 저자들은 기존 RLHF 파이프라인을 세 단계로 정리한다.  
첫째는 **지도 미세조정(SFT)**,  
둘째는 **선호 샘플링과 보상 학습**,  
셋째는 **강화학습을 통한 정책 최적화**다.

### SFT

처음에는 사전학습된 언어 모델을 다운스트림 과제의 고품질 데이터로 지도 미세조정해 기준이 되는 정책 \( \pi_{\mathrm{SFT}} \) 를 만든다. 요약이면 사람이 작성한 요약을, 대화면 더 도움이 되는 응답을 학습시켜 기본적인 형식과 분포를 잡는다. 이후의 모든 정렬 단계는 이 SFT 모델을 출발점으로 삼는다.

### Reward Modelling Phase

두 번째 단계에서는 프롬프트 \(x\) 에 대해 SFT 모델에서 두 개의 응답 \(y_1, y_2 \sim \pi_{\mathrm{SFT}}(y\mid x)\) 를 샘플링하고, 인간 라벨러가 둘 중 어느 응답을 더 선호하는지 표시한다. 선호된 응답을 \(y_w\), 비선호 응답을 \(y_l\) 로 표기한다. 논문은 이러한 선호가 관측 불가능한 잠재 보상 함수 \(r^*(x,y)\) 에 의해 생성된다고 가정한다.

가장 널리 쓰이는 선호 모델은 Bradley-Terry 모델이다. 이 모델에서는 두 응답 중 \(y_1\) 이 \(y_2\) 보다 선호될 확률이 두 보상의 지수화 비율로 주어진다.

$$
p^*(y_1 \succ y_2 \mid x)
=
\frac{\exp(r^*(x,y_1))}
{\exp(r^*(x,y_1)) + \exp(r^*(x,y_2))}
\tag{1}
$$

이 식은 보상 함수의 절대값이 아니라 **보상 차이** 만이 선호를 결정한다는 점을 보여 준다. 보상이 큰 응답일수록 선호될 가능성이 커지지만, 보상에 입력 \(x\) 만의 상수항을 더해도 확률은 바뀌지 않는다. 이 사실은 나중에 보상 동치류와 DPO 재매개변수화의 핵심이 된다.

정적 비교 데이터셋
\[
\mathcal D = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^N
\]
가 주어지면, 보상 모델 \(r_\phi(x,y)\) 는 최대우도추정으로 학습된다. 손실은 이진 분류 손실과 완전히 같은 꼴이다.

$$
\mathcal L_R(r_\phi,\mathcal D)
=
-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}
\Big[
\log \sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)
\Big]
\tag{2}
$$

즉, 보상 모델은 선호된 응답의 보상이 비선호 응답보다 높아지도록 학습된다. 실제 언어 모델 설정에서는 보상 네트워크를 완전히 새로 만드는 대신, SFT 모델 위에 선형 헤드를 올려 스칼라 보상을 예측하도록 초기화하는 경우가 많다. 또한 보상 분산을 줄이기 위해 프롬프트별 평균 보상을 0에 가깝게 맞추는 정규화를 적용하기도 한다.

### RL Fine-Tuning Phase

세 번째 단계에서는 학습된 보상 모델 \(r_\phi\) 를 사용해 실제 정책 \( \pi_\theta \) 를 최적화한다. 논문이 정리하는 표준 목적은 다음과 같다.

$$
\max_{\pi_\theta}
\;
\mathbb E_{x\sim\mathcal D,\; y\sim \pi_\theta(y\mid x)}
\big[r_\phi(x,y)\big]
-
\beta D_{\mathrm{KL}}
\big(
\pi_\theta(y\mid x) \,\|\, \pi_{\mathrm{ref}}(y\mid x)
\big)
\tag{3}
$$

여기서 \( \pi_{\mathrm{ref}} \) 는 기준 정책이며 대개 초기 SFT 모델과 같다. \( \beta \) 는 기준 정책으로부터 멀어지는 정도를 얼마나 강하게 억제할지 조절한다. 이 KL 항은 두 가지 역할을 한다. 하나는 보상 모델이 신뢰할 수 있는 분포로부터 정책이 너무 멀어지지 않도록 하는 것이고, 다른 하나는 고보상 응답 하나만 반복하는 모드 붕괴를 막는 것이다.

하지만 언어 생성은 이산적이어서 이 목적을 직접 미분하기 어렵고, 실제로는 PPO 같은 강화학습 알고리즘을 통해 근사적으로 최적화한다. 또한 실제 구현에서는
\[
r(x,y) = r_\phi(x,y) - \beta\big(\log \pi_\theta(y\mid x)-\log \pi_{\mathrm{ref}}(y\mid x)\big)
\]
같이 보상 안에 KL 보정을 묶어 넣고 PPO를 돌리는 방식이 널리 쓰인다. 이 지점에서 RLHF는 비로소 보상 학습과 정책 학습이라는 두 개의 별도 최적화 문제를 품게 되고, 이 복잡성이 DPO의 출발점이 된다.

## 4. Direct Preference Optimization

DPO의 출발점은 간단하다. 굳이 보상 모델을 명시적으로 학습한 뒤 RL로 정책을 찾을 필요가 있는가? 만약 식 (3)의 최적 정책을 보상 함수로부터 바로 쓸 수 있다면, 그 관계를 뒤집어서 정책을 직접 학습할 수 있다. 저자들은 바로 이 가능성을 이용한다.

### Deriving the DPO objective

식 (3)과 같은 KL 제약 보상 최대화 문제를 임의의 보상 함수 \(r(x,y)\) 에 대해 생각하면, 그 최적 정책은 다음 닫힌형태를 갖는다.

$$
\pi_r(y\mid x)
=
\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
\exp\!\left(\frac{1}{\beta}r(x,y)\right)
\tag{4}
$$

여기서 분할함수는
\[
Z(x)
=
\sum_y \pi_{\mathrm{ref}}(y\mid x)\exp\!\left(\frac{1}{\beta}r(x,y)\right)
\]
이다. 이 식은 KL 제약이 있는 보상 최적화의 해가 기준 정책에 대해 지수화된 보상으로 재가중한 형태임을 뜻한다. 보상이 높은 응답일수록 질량이 커지고, 기준 정책 확률이 0인 곳에는 질량이 새로 생기지 않는다.

식 (4)를 로그 형태로 다시 쓰면, 보상 함수를 정책의 함수로 표현할 수 있다.

$$
r(x,y)
=
\beta \log \frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+
\beta \log Z(x)
\tag{5}
$$

여기서 첫 번째 항은 정책이 기준 정책보다 특정 응답에 얼마나 더 많은 확률 질량을 두는지를 뜻한다. 두 번째 항 \( \beta\log Z(x) \) 는 프롬프트 \(x\) 만의 정규화 항이다. 중요하게도 Bradley-Terry 모델은 보상의 절대값이 아니라 두 응답 간 차이만을 사용한다. 따라서 식 (5)를 선호 확률에 대입하면 \( \log Z(x) \) 항이 정확히 소거된다.

그 결과, 최적 정책 \( \pi^* \) 가 만족해야 하는 인간 선호 확률은 다음과 같이 **정책 대 기준 정책의 로그비** 만으로 표현된다.

$$
p^*(y_1 \succ y_2 \mid x)
=
\frac{1}
{1+\exp\!\left(
\beta \log \frac{\pi^*(y_2\mid x)}{\pi_{\mathrm{ref}}(y_2\mid x)}
-
\beta \log \frac{\pi^*(y_1\mid x)}{\pi_{\mathrm{ref}}(y_1\mid x)}
\right)}
\tag{6}
$$

이제 더 이상 보상 모델을 따로 둘 필요가 없다. 데이터셋 \( \mathcal D \) 위에서 정책 \( \pi_\theta \) 자체를 최대우도로 학습하면 된다. 논문의 DPO 손실은 다음과 같다.

$$
\mathcal L_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}})
=
-\mathbb E_{(x,y_w,y_l)\sim \mathcal D}
\left[
\log \sigma\left(
\beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right)
\right]
\tag{7}
$$

이 손실은 겉으로 보면 간단한 이진 분류 목적이지만, 의미는 훨씬 풍부하다. 내부에 들어 있는 것은 단순 로그확률 차이가 아니라 **정책이 기준 정책에 비해 얼마나 더 선호 응답을 밀어주고 비선호 응답을 밀어내는가** 를 나타내는 로그비다. 즉 DPO는 "그냥 chosen을 많이 맞혀라"가 아니라 "reference에 비해 chosen의 상대적 우세를 키워라"를 학습한다. 이 구조 덕분에 기준 정책으로부터의 과도한 드리프트를 손실 안에 직접 흡수할 수 있다.

### What does the DPO update do?

논문은 DPO를 직관적으로 이해하기 위해 손실의 기울기를 전개한다. 먼저 암묵적 보상을
\[
\hat r_\theta(x,y)
=
\beta \log \frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\]
라고 두면, 기울기는 다음처럼 해석할 수 있다.

$$
\nabla_\theta \mathcal L_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}})
=
-\beta\,
\mathbb E_{(x,y_w,y_l)\sim\mathcal D}
\Big[
\sigma\big(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\big)
\big(
\nabla_\theta \log \pi_\theta(y_w\mid x)
-
\nabla_\theta \log \pi_\theta(y_l\mid x)
\big)
\Big]
$$

이 식은 세 층위로 읽힌다.  
첫째, 선호된 응답 \(y_w\) 의 로그확률을 올린다.  
둘째, 비선호 응답 \(y_l\) 의 로그확률을 내린다.  
셋째, 이 두 업데이트는 예제별 가중 \( \sigma(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)) \) 로 조절된다.

가중항의 의미가 중요하다. 현재 정책이 이미 \(y_w\) 를 \(y_l\) 보다 충분히 더 높게 평가한다면, 즉 암묵적 보상이 이미 올바른 순서를 만들고 있다면 이 값은 작아진다. 반대로 아직 \(y_l\) 쪽을 더 높게 평가하고 있다면 이 값이 커져서 업데이트가 강해진다. 따라서 DPO는 모든 선호쌍을 똑같이 밀어붙이지 않고, 아직 잘못 정렬된 쌍을 우선적으로 수정한다. 저자들이 Table 3에서 보여 주는 것처럼, 이 가중이 없는 순진한 unlikelihood 류 목적은 응답 품질을 쉽게 붕괴시킬 수 있다.

### DPO outline

실제 절차는 간단하다. 먼저 기준 정책 \( \pi_{\mathrm{ref}} \) 로부터 각 프롬프트에 대한 여러 후보 응답을 샘플링하고, 인간이 그중 어느 응답을 더 선호하는지 라벨링해 오프라인 데이터셋
\[
\mathcal D = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^N
\]
를 만든다. 그다음 정책 \( \pi_\theta \) 를 식 (7)을 최소화하도록 학습한다. 이것으로 끝이다. 보상 모델도 없고, on-policy RL 루프도 없으며, value function도 필요 없다.

실전에서는 공개 선호 데이터셋을 재사용하는 경우가 많다. 이런 데이터셋이 원래 \( \pi_{\mathrm{SFT}} \) 에서 샘플된 것이라면 기준 정책을 \( \pi_{\mathrm{ref}}=\pi_{\mathrm{SFT}} \) 로 두면 된다. 만약 원래의 SFT 모델이 없으면, 저자들은 선호된 응답 \( (x,y_w) \) 만으로 간단한 지도학습을 하여 reference를 초기화한다. 즉

\[
\pi_{\mathrm{ref}}
=
\arg\max_\pi \mathbb E_{x,y_w\sim\mathcal D}\big[\log \pi(y_w\mid x)\big]
\]

로 두어, 실제 선호 데이터가 생성된 분포와 현재 reference 사이의 어긋남을 줄인다. 이렇게 만들어진 reference는 DPO 손실에서 비교의 기준점 역할을 한다. 부록 B에서는 이 손실을 PyTorch로 구현하는 아주 짧은 코드와 기본 하이퍼파라미터를 제시한다.

## 5. Theoretical Analysis of DPO

이 절은 DPO가 단지 경험적으로 잘 되는 간단한 트릭이 아니라, 보상 함수와 정책 사이의 관계를 정확히 이용한 재매개변수화라는 점을 보이는 부분이다. 논문은 먼저 "보상 함수는 입력별 상수항까지 포함하면 본질적으로 동치류로만 구분된다"는 사실을 정리하고, 그 위에서 DPO가 각 동치류마다 유일한 대표 보상 함수를 선택한다고 설명한다. 이어서 PPO 같은 actor-critic 계열 RLHF가 왜 불안정해지기 쉬운지도 같은 틀에서 해석한다.

### 5.1 Your Language Model Is Secretly a Reward Model

DPO의 핵심 문장은 제목 그대로다. **언어 모델은 이미 보상 모델을 암묵적으로 품고 있다.**  
구체적으로, 기준 정책 \( \pi_{\mathrm{ref}} \) 가 정해져 있을 때 정책 \( \pi \) 는 다음 형태의 암묵적 보상을 정의한다.

\[
\hat r_\pi(x,y)
=
\beta \log \frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\]

즉, 어떤 응답이 기준 정책보다 얼마나 더 많이 선택되도록 재분배되었는지가 곧 그 응답의 보상처럼 작동한다. 이 해석을 이론적으로 정당화하기 위해 논문은 보상 함수 사이의 동치관계를 먼저 정의한다.

**Definition 1.** 두 보상 함수 \(r(x,y)\) 와 \(r'(x,y)\) 가 어떤 함수 \(f(x)\) 에 대해

\[
r(x,y) - r'(x,y) = f(x)
\]

를 만족하면 두 보상 함수는 동치라고 한다.

이 정의는 프롬프트 \(x\) 에만 의존하는 상수 이동은 응답 간의 상대적 비교를 바꾸지 않는다는 뜻이다. 한 프롬프트 아래에서 가능한 모든 응답에 같은 값을 더해도, 어떤 응답이 더 높은지라는 순서 자체는 바뀌지 않는다.

**Lemma 1.** Plackett-Luce 계열, 특히 Bradley-Terry 선호 모델 아래에서는 같은 동치류에 속한 두 보상 함수가 같은 선호 분포를 유도한다.

이 결과는 선호 확률이 보상 차이 또는 비율에만 의존하기 때문에 성립한다. 각 보상에 \(f(x)\) 를 더하면 분자와 분모에 동일한 배수가 생기지만, 최종 확률에서는 그 배수가 정확히 상쇄된다. 따라서 선호 데이터만으로는 보상 함수의 절대 수준을 식별할 수 없고, 동치류까지만 식별할 수 있다.

**Lemma 2.** 같은 동치류에 속한 두 보상 함수는 KL 제약 보상 최대화 문제에서 같은 최적 정책을 유도한다.

이 역시 식 (4)에서 바로 보인다. 보상에 \(f(x)\) 를 더하면 분자 전체에 \( \exp(f(x)/\beta) \) 가 곱해지고, 분할함수에도 정확히 같은 항이 곱해지므로 최종 정책은 변하지 않는다. 다시 말해 선호 모델과 RL 최적화 문제는 모두 보상 함수의 절대 레벨에 무감하다.

이제 논문은 DPO 재매개변수화가 표현력을 잃지 않는다는 핵심 정리를 제시한다.

**Theorem 1.** 적절한 가정 아래에서, Plackett-Luce 계열 선호 모델과 일관적인 모든 보상 동치류는 어떤 정책 \( \pi(y\mid x) \) 와 기준 정책 \( \pi_{\mathrm{ref}}(y\mid x) \) 에 대해

\[
r(x,y)
=
\beta \log \frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\]

형태를 갖는 대표 원소로 표현될 수 있다.

저자들의 증명 아이디어는 다음과 같다. 임의의 보상 함수 \(r(x,y)\) 가 주어지면, 식 (4)에 따라 그 보상에 대응하는 최적 정책 \( \pi_r \) 가 존재한다. 이때 보상 함수에서 분할함수의 로그를 빼는 사영 연산

$$
f(r;\pi_{\mathrm{ref}},\beta)(x,y)
=
r(x,y)
-
\beta \log \sum_y \pi_{\mathrm{ref}}(y\mid x)
\exp\!\left(\frac{1}{\beta}r(x,y)\right)
\tag{8}
$$

를 정의하면, 이 새로운 보상은 원래 보상과 같은 동치류에 속하면서 동시에

\[
f(r;\pi_{\mathrm{ref}},\beta)(x,y)
=
\beta \log \frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\]

형태를 만족한다. 즉, DPO가 사용하는 보상 형태는 임의의 보상 동치류에 대해 적어도 하나의 대표자를 항상 제공한다.

논문은 이 정리를 다른 각도에서도 해석한다. DPO는 각 동치류에서 다음 조건을 만족하는 보상 함수를 선택한다.

$$
\sum_y \pi_{\mathrm{ref}}(y\mid x)\exp\!\left(\frac{1}{\beta}r(x,y)\right) = 1
\tag{9}
$$

이 식은 단순한 정규화 조건이지만 의미는 크다. 식 (4)에 따르면 좌변은 사실 그 보상에 대응하는 최적 정책의 분할함수다. DPO는 바로 이 분할함수를 1로 맞추는 대표 보상 함수를 선택함으로써, 보상과 정책 사이의 관계를 완전히 닫힌형태로 만든다. 따라서 DPO가 다루는 보상 함수 공간은 원래 RLHF가 다루던 보상 동치류 공간과 본질적으로 같다.

이 절이 중요한 이유는 두 가지다.  
첫째, DPO가 "정책을 직접 학습하니까 보상 모델의 일반성을 잃는 것 아니냐"는 의문에 대해 아니라고 답한다.  
둘째, 보상 모델의 under-specification이 오히려 DPO를 가능하게 만드는 조건임을 보여 준다. 선호 데이터가 어차피 보상의 절대값을 정하지 못하기 때문에, DPO는 그 남는 자유도를 이용해 최적 정책을 직접 표현하는 형태를 택한다.

### 5.2 Instability of Actor-Critic Algorithms

저자들은 같은 틀을 이용해 PPO류 RLHF의 불안정성도 해석한다. 핵심은 표준 RLHF가 최적화하는 목적을 다시 쓰면, 사실상 DPO와 동등한 보상 클래스 위에서 학습하지만 그 과정에서 정규화 항을 별도로 다뤄야 한다는 점이다.

논문은 파라미터화된 보상 모델 \(r_\phi(x,y)\) 가 주어졌을 때, 최적 정책 \( \pi^* \) 를 향한 KL 최소화 관점에서 다음 목적이 나온다고 설명한다.

$$
\max_{\pi_\theta}
\;
\mathbb E_{\pi_\theta(y\mid x)}
\left[
r_\phi(x,y)
-
\beta \log \sum_y \pi_{\mathrm{ref}}(y\mid x)
\exp\!\left(\frac{1}{\beta}r_\phi(x,y)\right)
-
\beta \log \frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\right]
\tag{10}
$$

여기서 두 번째 항은 보상 정규화에 해당하며, 저자들은 이를 \(f(r_\phi,\pi_{\mathrm{ref}},\beta)\) 로 본다. 이 항은 최적해 자체를 바꾸지는 않지만, 정책 기울기 분산에는 큰 영향을 준다. 만약 이 항을 무시하면 보상 규모가 들쑥날쑥해져 학습이 불안정해질 수 있다. PPO류 방법에서는 이를 value function으로 보정하거나, 사람의 기준 응답 하나를 사용한 baseline으로 근사하거나, reward normalization을 따로 넣는 식으로 해결한다. 그러나 이 모든 장치는 추가 추정 오차와 튜닝 난이도를 낳는다.

DPO는 이 문제를 다른 방식으로 피한다. 보상을 직접 학습해 RL로 최적화하지 않고, 애초에 정규화된 대표 보상 함수만을 허용하는 재매개변수화를 사용하기 때문이다. 따라서 baseline이나 value function이 없어도 되고, 보상 정규화 때문에 생기는 고분산 정책 기울기 문제도 훨씬 약해진다. 이 절의 메시지는 단순하다. **DPO가 단순한 이유는 RL 문제를 억지로 쉬운 문제로 바꿔서가 아니라, 원래 문제의 해를 직접 매개변수화했기 때문이다.**

[Figure 2 원문 이미지 및 원문 캡션 삽입]

## 6. Experiments

실험 절의 질문은 세 가지다.  
첫째, DPO는 KL 제약 보상 최대화라는 RLHF의 원래 목적을 얼마나 잘 최적화하는가.  
둘째, 실제 인간 선호 데이터셋에서도 PPO 수준으로 확장되는가.  
셋째, GPT-4를 자동 평가자로 사용해도 인간 평가와 충분히 일치하는가.

### Tasks

논문은 세 가지 개방형 텍스트 생성 과제를 사용한다.

첫 번째는 **IMDb 감성 제어 생성** 이다. 입력 \(x\) 는 IMDb 영화 리뷰의 접두부이고, 모델은 긍정 감성을 띠는 이어쓰기 \(y\) 를 생성해야 한다. 이 실험에서는 사전학습된 감성 분류기가 실제 보상 함수를 대신하므로, 선호쌍도 분류기 점수에 따라 자동으로 만든다. 따라서 보상-KL frontier를 정밀하게 측정할 수 있는 통제된 실험이 된다.

두 번째는 **Reddit TL;DR 요약** 이다. 입력은 Reddit 포스트, 출력은 포스트의 핵심을 요약한 TL;DR이다. 사람 선호 데이터는 Stiennon 등의 요약 선호 데이터셋을 사용한다. 자동 평가 지표인 ROUGE는 인간 선호와 잘 맞지 않는 경우가 많기 때문에, 이 논문은 GPT-4를 비교 평가자로 사용한다.

세 번째는 **Anthropic Helpful and Harmless 단일 턴 대화** 다. 입력은 사용자 질의 하나이고, 출력은 유용하고 해롭지 않은 응답이다. 이 데이터셋에는 사람-모델 대화 맥락과 함께 두 후보 응답에 대한 선호 라벨이 포함되어 있다. 이 과제는 요약보다 훨씬 넓은 의미의 도움됨과 안전성을 평가하게 해 준다.

### Evaluation

평가 방식도 과제에 따라 다르다. 감성 제어 실험에서는 실제 보상 함수가 알려져 있으므로, 각 방법이 달성한 **기대 보상과 reference로부터의 KL** 을 직접 측정해 frontier를 비교한다. 이는 RLHF 목적 자체를 얼마나 잘 푸는지를 가장 정직하게 보여 준다.

반면 요약과 대화에서는 참 보상 함수가 없으므로, 저자들은 **win rate** 를 사용한다. 요약에서는 테스트셋의 인간 작성 요약을 기준 응답으로 두고, 대화에서는 Anthropic HH 테스트셋의 chosen 응답을 기준으로 둔다. 각 방법이 생성한 응답과 기준 응답을 GPT-4에 비교시켜 더 나은 쪽을 고르게 하고, 그 비율을 승률로 계산한다. 논문은 GPT-4 평가가 믿을 만한지 따로 인간 평가까지 수행하며, 그 결과는 6.4절과 부록 D.3에 제시한다.

[Figure 3 원문 이미지 및 원문 캡션 삽입]

### Methods

비교 대상은 꽤 폭넓다. 가장 기본적인 기준선은 SFT 모델 자체다. 여기에 선호된 응답만으로 지도학습하는 **Preferred-FT**, 선호 응답 확률은 올리고 비선호 응답 확률은 내리는 **Unlikelihood**, 표준 RLHF의 대표인 **PPO**, 참 보상 함수를 아는 오라클 버전인 **PPO-GT**, 그리고 reward model로 점수 매긴 샘플 중 최고점을 택하는 **Best of N** 이 포함된다. 요약에서는 GPT-J 제로샷 프롬프팅도, 대화에서는 Pythia-2.8B의 2-shot prompting도 비교한다.

Best of N은 계산량이 매우 크지만 강한 기준선이다. 샘플을 많이 뽑을수록 reward model이 좋아하는 응답을 찾을 가능성이 높기 때문이다. 따라서 만약 DPO가 Best of N 수준에 근접한다면, RL 없이도 꽤 강력한 정책을 만들 수 있다는 의미가 된다.

### 6.1 How well can DPO optimize the RLHF objective?

이 절에서 저자들은 가장 공정한 질문을 던진다. DPO와 PPO는 결국 같은 KL 제약 보상 최대화 문제를 풀고자 한다. 그렇다면 실제로 어느 쪽이 더 효율적인가?

결과는 Figure 2의 왼쪽 그래프에 정리된다. 여러 하이퍼파라미터 설정으로 PPO, PPO-GT, DPO, Unlikelihood, Preferred-FT를 학습시키고, 각 체크포인트마다 기대 보상과 sequence-level KL을 측정해 frontier를 그린다. 여기서 중요한 것은 보상만 높다고 좋은 것이 아니라, 같은 KL 수준에서 더 높은 보상을 얻는지가 핵심이라는 점이다.

논문의 결론은 명확하다. **DPO의 frontier가 PPO를 엄격하게 지배한다.** 즉, 같은 KL 수준에서 더 높은 보상을 얻고, 같은 보상을 달성할 때 더 작은 KL로 유지된다. 심지어 PPO가 참 보상 함수를 직접 보는 오라클 설정(PPO-GT)과 비교해도 DPO가 더 나은 frontier를 보인다. 저자들이 강조하는 대목은 바로 이것이다. DPO와 PPO는 이론적으로 같은 목적을 푼다고 말할 수 있지만, 실제 최적화 효율은 상당히 다르다. RL-free 구조가 오히려 더 직접적이고 안정적인 최적화를 만들어 낸다는 의미다.

### 6.2 Can DPO scale to real preference datasets?

이 절에서는 더 현실적인 두 과제, 즉 TL;DR 요약과 Anthropic HH 대화로 넘어간다.

요약 실험에서 DPO, PPO, Preferred-FT는 같은 GPT-J 기반 SFT 모델을 출발점으로 사용한다. 생성 온도를 0.0부터 1.0까지 바꾸며 GPT-4 승률을 측정한 결과, DPO는 **온도 0.0에서 약 61% 승률** 을 기록해 PPO의 최고 성능인 **약 57%** 를 넘어선다. 더욱 중요한 차이는 강건성이다. PPO는 샘플링 온도가 올라가면 성능이 급격히 흔들리지만, DPO는 온도 변화에 더 안정적이다. 논문은 \(\beta\) 를 크게 튜닝하지 않았기 때문에, 이 결과가 오히려 DPO 잠재력을 과소평가할 수 있다고 덧붙인다.

대화 실험에서는 상황이 조금 다르다. 표준 SFT 모델이 따로 주어지지 않으므로, 저자들은 먼저 선호된 응답만으로 Preferred-FT를 수행해 reference를 만든 뒤 DPO를 학습한다. 비교 대상에는 Best of 128, base Pythia-2.8B, 2-shot prompting, 공개 PPO-HH 모델이 들어간다. 결과적으로 DPO는 테스트셋의 chosen 응답을 이기는 유일한 계산 효율적 방법이며, 매우 비싼 Best of 128과 비슷하거나 더 나은 수준에 도달한다. 또한 Figure 3의 오른쪽 그래프는 DPO가 학습 초반부터 비교적 안정적으로 성능을 끌어올리며, 온도 0.7과 1.0 모두에서 chosen 대비 우위를 유지한다는 점을 보여 준다.

### 6.3 Generalization to a new input distribution

저자들은 TL;DR 요약에서 학습한 PPO와 DPO 정책을 전혀 다른 입력 분포인 CNN/DailyMail 뉴스 기사에 옮겨 본다. 이는 "보상 모델을 명시적으로 두지 않으면 분포 이동에 약하지 않겠느냐"는 자연스러운 의문에 대한 초기 검증이다.

[Table 1 원문 표 및 원문 캡션 삽입]

Table 1에서 DPO는 CNN/DailyMail 테스트셋에서도 PPO보다 일관되게 높은 승률을 보인다. 온도 0에서는 DPO가 0.36, PPO가 0.26이고, 온도 0.25에서는 DPO가 0.31, PPO가 0.23이다. 절대 승률 자체는 높지 않지만, 포인트는 두 가지다. 첫째, Reddit TL;DR에서 학습한 선호가 뉴스 요약이라는 다른 도메인으로 완전히 무너지지 않는다. 둘째, DPO는 PPO가 학습 중 사용한 추가 unlabeled prompt 없이도 이 일반화 실험에서 더 강한 성능을 유지한다. 논문은 이를 DPO가 PPO만큼은 일반화할 수 있다는 초기 증거로 해석한다.

### 6.4 Validating GPT-4 judgments with human judgments

자동 평가자로 GPT-4를 쓰려면, 그 판단이 인간 판단과 어느 정도 일치하는지 확인해야 한다. 저자들은 TL;DR 요약 과제에서 인간 평가 실험을 수행하고 두 종류의 GPT-4 프롬프트를 비교한다. 하나는 단순히 "어느 요약이 더 중요한 포인트를 잘 요약하는가"만 묻는 GPT-4 (S)이고, 다른 하나는 "중요한 포인트를 잘 요약하면서 불필요한 세부를 넣지 않는가, 즉 정확하면서도 간결한가"를 묻는 GPT-4 (C)다.

[Table 2 원문 표 및 원문 캡션 삽입]

비교는 세 조합에서 이뤄진다. 성능이 높은 DPO(temperature 0.25), 중간 정도의 SFT(temperature 0.25), 성능이 낮은 PPO(temperature 1.0)를 각각 기준 PPO(temperature 0)와 맞붙인다. Table 2를 보면 DPO는 인간 평가에서 58% 승률을 얻고, GPT-4 (C)는 54%, GPT-4 (S)는 47%를 준다. SFT는 인간 43%, GPT-4 (C) 32%, GPT-4 (S) 27%이며, PPO-1은 인간 17%, GPT-4 (C) 12%, GPT-4 (S) 13%다. 즉 절대값이 완전히 같지는 않지만, 좋은 방법과 나쁜 방법의 순서를 잡아내는 방향은 인간과 꽤 잘 맞는다.

또한 인간-인간 합의율과 GPT-4-인간 합의율을 비교하면, GPT-4가 인간 두 명 사이의 일치도와 비슷한 수준으로 판단하는 경우가 많다. 이 때문에 저자들은 GPT-4를 완전한 대체재가 아니라 **합리적인 프록시 평가자** 로 본다. 특히 간결성까지 묻는 GPT-4 (C) 프롬프트가 사람 취향과 더 비슷하다고 판단해, 본문의 주 결과는 이 프롬프트를 사용한다.

## 7. Discussion

논문의 결론은 비교적 분명하다. 선호 학습은 강력한 정렬 프레임워크이고, DPO는 이 프레임워크를 훨씬 단순한 학습 절차로 구현한다. 보상 모델을 따로 학습하고 RL 루프를 돌리는 대신, 정책-보상 대응 관계를 직접 이용해 선호 데이터를 분류 문제로 바꾸면, 이론적 목적은 그대로 유지하면서도 학습은 더 단순하고 안정적으로 만들 수 있다는 것이 이 논문의 핵심 메시지다.

저자들은 동시에 몇 가지 한계를 남긴다. 첫째, DPO 정책의 out-of-distribution 일반화는 아직 초기 결과만 제시되었을 뿐이며 더 넓은 검증이 필요하다. 둘째, self-labeling이나 unlabeled prompt 활용 같은 확장 기법을 DPO에서도 똑같이 사용할 수 있는지 아직 분명하지 않다. 셋째, Figure 3 오른쪽에서 보이는 약간의 성능 하락이 reward over-optimization의 징후인지도 더 연구가 필요하다. 넷째, 실험은 최대 6B 규모까지 수행되었지만, 훨씬 큰 최신 모델에서도 같은 장점이 유지되는지는 열려 있다. 마지막으로 GPT-4 평가의 프롬프트 민감성, 그리고 텍스트 이외의 생성 모델로의 확장도 중요한 후속 과제다.

## Acknowledgements

저자들은 Knight-Hennessy Graduate Fellowship, CIFAR, Stanford SAL 및 HAI의 Generative AI for the Future of Learning seed grant, Stanford CRFM의 컴퓨트 지원, 그리고 ONR grant N00014-20-1-2675의 지원을 받았다고 밝힌다. 즉, 이 연구는 이론 연구이면서 동시에 꽤 큰 규모의 실험 자원이 필요했던 프로젝트다.

## References

이 논문이 의존하는 참고문헌 축은 크게 다섯 가지다.  
첫째, RLHF와 인간 선호 학습의 고전적 계보.  
둘째, Bradley-Terry와 Plackett-Luce 같은 순위·선호 모델.  
셋째, PPO와 KL-regularized RL, control-as-inference 계열의 최적화 이론.  
넷째, instruction tuning과 대규모 언어 모델 정렬 실험.  
다섯째, TL;DR 요약, Anthropic HH, GPT-4 자동 평가처럼 본문 실험을 구성하는 데이터셋과 평가 연구다.  
원문은 총 51편의 참고문헌을 인용한다.

## Author Contributions

저자들은 전원이 실험 설계, 분석, 반복 개선, 원고 작성과 편집, 프로젝트 진행 관리에 기여했다고 밝힌다.

Rafael Rafailov는 자가회귀 보상 모델 아이디어를 바탕으로 DPO 목적함수를 유도했고, 알고리즘의 이론적 성질을 증명했으며, 해당 수학 절과 부록을 주도적으로 작성했다. 또한 실험 구성과 PPO 및 reward learning 기준선 일부에도 기여했다.

Archit Sharma는 weighted regression을 PPO의 대안으로 보는 초기 문제의식을 제시했고, unlikelihood와의 연결, DPO 구현과 기준선 설계, 초기 탐색 실험, 감성 제어와 요약 실험 조직, GPT-4 평가 설계 개선, 본문 방법·실험 부분 작성에 큰 비중을 차지했다.

Eric Mitchell은 초기 DPO 구현과 첫 실험을 수행했고, 대규모 요약 및 대화 DPO 모델 학습, GPT-4 승률 평가 인프라, 인간 평가 실험 운영과 분석, 초록·서론·관련연구·논의·실험 작성에 크게 기여했다.

Chelsea Finn, Christopher Manning, Stefano Ermon은 연구를 지도하고, 아이디어와 실험을 제안했으며, 논문 작성 전반을 지원했다.

## A. Mathematical Derivations

이 부록은 본문 4절과 5절에서 사용한 수식들을 더 천천히 전개한 부분이다. 본문에서는 핵심 결론만 제시했지만, 여기서는 식 (4), (6), DPO 기울기, Lemma와 Theorem의 증명이 차례대로 나온다.

### A.1 Deriving the Optimum of the KL-Constrained Reward Maximization Objective

먼저 본문 식 (4)를 유도한다. 출발점은 식 (3)과 같은 KL 제약 보상 최대화 문제다. 논문은 이를 비모수 정책 클래스에 대해 다음처럼 적는다.

$$
\max_{\pi}
\;
\mathbb E_{x\sim\mathcal D,\; y\sim\pi}
\big[r(x,y)\big]
-
\beta D_{\mathrm{KL}}\big(\pi(y\mid x)\,\|\,\pi_{\mathrm{ref}}(y\mid x)\big)
\tag{11}
$$

이 목적을 전개하면

$$
\begin{aligned}
\max_{\pi}\;&
\mathbb E_{x\sim\mathcal D,\; y\sim\pi(y\mid x)}
\left[
r(x,y) - \beta \log \frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\right] \\
=\;&
\min_{\pi}
\mathbb E_{x\sim\mathcal D,\; y\sim\pi(y\mid x)}
\left[
\log \frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} - \frac{1}{\beta}r(x,y)
\right]
\end{aligned}
$$

가 된다. 여기서 논문은 분할함수

\[
Z(x)=\sum_y \pi_{\mathrm{ref}}(y\mid x)\exp\!\left(\frac{1}{\beta}r(x,y)\right)
\]

를 도입하고, 위 식을

$$
\min_{\pi}
\mathbb E_{x\sim\mathcal D,\; y\sim\pi(y\mid x)}
\left[
\log
\frac{\pi(y\mid x)}
{\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\exp(\frac{1}{\beta}r(x,y))}
-
\log Z(x)
\right]
\tag{12}
$$

로 바꾼다. 이제

\[
\pi^*(y\mid x)
=
\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
\exp\!\left(\frac{1}{\beta}r(x,y)\right)
\]

를 정의하면, 목적은

$$
\min_{\pi}
\mathbb E_{x\sim\mathcal D}
\left[
\mathbb E_{y\sim\pi(y\mid x)}
\left[
\log \frac{\pi(y\mid x)}{\pi^*(y\mid x)}
\right]
-
\log Z(x)
\right]
\tag{13}
$$

가 되고, 다시

$$
\min_{\pi}
\mathbb E_{x\sim\mathcal D}
\Big[
D_{\mathrm{KL}}(\pi(y\mid x)\,\|\,\pi^*(y\mid x))
-
\log Z(x)
\Big]
\tag{14}
$$

로 정리된다. 여기서 \(Z(x)\) 는 정책 \(\pi\) 와 무관하므로, 목적은 결국 KL 항을 최소화하는 문제로 바뀐다. Gibbs 부등식에 의해 KL은 두 분포가 같을 때에만 0이므로 최적 정책은

$$
\pi(y\mid x)
=
\pi^*(y\mid x)
=
\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
\exp\!\left(\frac{1}{\beta}r(x,y)\right)
\tag{15}
$$

가 된다. 이것이 본문 식 (4)의 엄밀한 유도다. 이 부록이 보여 주는 핵심은 KL 제약 보상 최대화의 해가 사실상 **reference 정책의 지수적 틸팅(exponential tilting)** 이라는 점이다.

### A.2 Deriving the DPO Objective Under the Bradley-Terry Model

다음은 본문 식 (6)과 식 (7)의 유도다. Bradley-Terry 모델은

$$
p^*(y_1 \succ y_2 \mid x)
=
\frac{\exp(r^*(x,y_1))}
{\exp(r^*(x,y_1))+\exp(r^*(x,y_2))}
\tag{16}
$$

로 주어진다. 본문 4절에서 이미 잠재 보상은 최적 정책을 통해

$$
r^*(x,y)
=
\beta \log \frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+
\beta \log Z(x)
\tag{17}
$$

로 쓸 수 있음을 보였다. 이 식을 식 (16)에 대입하면, 분자와 분모에 공통으로 들어가는 \( \exp(\beta\log Z(x)) \) 항이 모두 지워진다. 남는 것은 두 응답에 대한 정책/기준정책 로그비의 차이뿐이다.

즉,

\[
p^*(y_1 \succ y_2 \mid x)
=
\sigma\!\left(
\beta \log \frac{\pi^*(y_1\mid x)}{\pi_{\mathrm{ref}}(y_1\mid x)}
-
\beta \log \frac{\pi^*(y_2\mid x)}{\pi_{\mathrm{ref}}(y_2\mid x)}
\right)
\]

가 된다. 이제 \(\pi^*\) 를 파라미터화된 정책 \(\pi_\theta\) 로 바꾸고 최대우도 학습을 하면, 본문 식 (7)의 DPO 손실이 바로 얻어진다. 이 부록의 메시지는 짧다. **Bradley-Terry가 보상 차이만 사용하기 때문에, 분할함수는 처음부터 최적화 문제에 남지 않는다.**

### A.3 Deriving the DPO Objective Under the Plackett-Luce Model

논문은 DPO가 Bradley-Terry 쌍대 비교에만 한정되지 않음을 보여 주기 위해 더 일반적인 Plackett-Luce 순위 모델도 전개한다. \(K\)개의 후보 응답 \(y_1,\dots,y_K\) 가 있고 사용자가 그 순위를 나타내는 순열 \(\tau:[K]\to[K]\) 를 준다고 하자. 그러면 Plackett-Luce 모델은

$$
p^*(\tau \mid y_1,\dots,y_K,x)
=
\prod_{k=1}^{K}
\frac{\exp(r^*(x,y_{\tau(k)}))}
{\sum_{j=k}^{K}\exp(r^*(x,y_{\tau(j)}))}
\tag{18}
$$

로 쓰인다. \(K=2\) 이면 이는 Bradley-Terry와 같다. 여기에도 본문 식 (5)와 동일한 재매개변수화를 대입할 수 있다. 마찬가지로 각 항의 분자와 분모에서 \(Z(x)\) 가 소거되므로,

$$
p^*(\tau \mid y_1,\dots,y_K,x)
=
\prod_{k=1}^{K}
\frac{
\exp\!\left(
\beta \log \frac{\pi^*(y_{\tau(k)}\mid x)}{\pi_{\mathrm{ref}}(y_{\tau(k)}\mid x)}
\right)
}{
\sum_{j=k}^{K}
\exp\!\left(
\beta \log \frac{\pi^*(y_{\tau(j)}\mid x)}{\pi_{\mathrm{ref}}(y_{\tau(j)}\mid x)}
\right)
}
\tag{19}
$$

가 된다. 그러면 순위 데이터셋
\[
\mathcal D = \{(\tau^{(i)}, y_1^{(i)},\dots,y_K^{(i)},x^{(i)})\}_{i=1}^N
\]
가 있을 때, 정책에 대한 최대우도 목적은

$$
\mathcal L_{\mathrm{DPO}}(\pi_\theta,\pi_{\mathrm{ref}})
=
-
\mathbb E_{\tau,y_1,\dots,y_K,x\sim\mathcal D}
\left[
\log
\prod_{k=1}^{K}
\frac{
\exp\!\left(
\beta \log \frac{\pi_\theta(y_{\tau(k)}\mid x)}{\pi_{\mathrm{ref}}(y_{\tau(k)}\mid x)}
\right)
}{
\sum_{j=k}^{K}
\exp\!\left(
\beta \log \frac{\pi_\theta(y_{\tau(j)}\mid x)}{\pi_{\mathrm{ref}}(y_{\tau(j)}\mid x)}
\right)
}
\right]
\tag{20}
$$

가 된다. 요점은 pairwise preference가 아닌 ranking supervision에도 같은 아이디어가 그대로 확장된다는 것이다. DPO의 본질은 "보상 대신 정책 로그비를 쓰는 것"이지, 비교가 두 개여야 한다는 데 있지 않다.

### A.4 Deriving the Gradient of the DPO Objective

본문에서 직관적으로 설명한 기울기를 부록에서는 계산으로 보여 준다. 출발점은 식 (7)을 다시 쓴 다음 식이다.

$$
\nabla_\theta \mathcal L_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}})
=
-
\nabla_\theta
\mathbb E_{(x,y_w,y_l)\sim\mathcal D}
\left[
\log \sigma\left(
\beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
-
\beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
\right)
\right]
\tag{21}
$$

여기서

\[
u
=
\beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
-
\beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
\]

라고 두면,

$$
\nabla_\theta \mathcal L_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}})
=
-
\mathbb E_{(x,y_w,y_l)\sim\mathcal D}
\left[
\frac{\sigma'(u)}{\sigma(u)} \nabla_\theta u
\right]
\tag{22}
$$

가 된다. 이제 \(\sigma'(x)=\sigma(x)(1-\sigma(x))\), \(\sigma(-x)=1-\sigma(x)\) 를 이용하면

\[
\frac{\sigma'(u)}{\sigma(u)}
=
1-\sigma(u)
=
\sigma(-u)
\]

이므로 최종적으로

\[
\nabla_\theta \mathcal L_{\mathrm{DPO}}
=
-
\mathbb E
\left[
\beta
\sigma\left(
\beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right)
\big(
\nabla_\theta \log \pi_\theta(y_w\mid x)
-
\nabla_\theta \log \pi_\theta(y_l\mid x)
\big)
\right]
\]

를 얻는다. 그리고 여기에

\[
\hat r_\theta(x,y)
=
\beta \log \frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\]

를 대입하면 본문에서 제시한 해석 가능한 형태가 나온다. 이 전개는 DPO가 선호된 응답의 likelihood를 키우고 비선호 응답의 likelihood를 줄이는 동시에, 현재 정렬 상태에 따라 예제별 가중을 다르게 준다는 사실을 수식으로 확인해 준다.

### A.5 Proof of Lemma 1 and 2

부록은 5.1절의 Lemma들을 아주 직접적으로 증명한다.

**Lemma 1 restated.** Plackett-Luce, 특히 Bradley-Terry 프레임워크 아래에서 같은 동치류의 보상 함수는 같은 선호 분포를 만든다.

증명 아이디어는 단순하다. \(r'(x,y)=r(x,y)+f(x)\) 라고 하자. 그러면 Plackett-Luce의 각 선택 확률 항에서 분자와 분모 모두에 \( \exp(f(x)) \) 가 곱해진다. 따라서 각 항에서 그 값이 상쇄되고, 전체 순위 확률도 완전히 동일하게 남는다. Bradley-Terry는 Plackett-Luce의 특수한 경우이므로 자동으로 포함된다.

**Lemma 2 restated.** 같은 동치류의 보상 함수는 KL 제약 RL 문제에서 같은 최적 정책을 만든다.

여기서는 식 (4)에 \(r'(x,y)=r(x,y)+f(x)\) 를 대입하면 된다. 분자에 \( \exp(f(x)/\beta) \) 가 생기고, 분할함수 \(Z(x)\) 에도 동일한 값이 곱해진다. 따라서 결과 정책은 \(\pi_{r'}(y\mid x)=\pi_r(y\mid x)\) 가 된다. 즉, 선호 분포도 같고 최적 정책도 같으므로, 실제 학습에 중요한 것은 개별 보상 함수가 아니라 보상 동치류다.

### A.6 Proof of Theorem 1

이 부분은 Theorem 1을 좀 더 엄밀하게 적는다.

**Theorem 1 restated.** 기준 정책이 모든 \(x,y\) 쌍에 대해 \( \pi_{\mathrm{ref}}(y\mid x) > 0 \) 이고 \( \beta > 0 \) 일 때, 5절에서 정의한 모든 보상 동치류는 어떤 정책 \(\pi(y\mid x)\) 에 대해

\[
r(x,y)=\beta \log \frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\]

형태로 표현될 수 있다.

증명은 본문에서 본 것처럼 임의의 보상 \(r(x,y)\) 를 택한 뒤, 그 최적 정책 \(\pi_r\) 를 식 (4)로 정의하는 데서 시작한다. 식 (5)에 따라

\[
r(x,y)
=
\beta \log \frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+
\beta \log Z(x)
\]

이고, 따라서

\[
r'(x,y)
=
r(x,y)-\beta\log Z(x)
\]

를 정의하면 \(r'\) 는 \(r\) 와 같은 동치류에 속하면서 곧바로

\[
r'(x,y)
=
\beta \log \frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\]

형태가 된다. 이것이 Theorem 1의 핵심이다.

논문은 여기서 더 나아가, 각 동치류에서 이런 재매개변수화를 가지는 보상 함수가 **유일하다** 는 Proposition 1도 제시한다.  
가정은 동일하다. 어떤 동치류 안에 두 보상 함수 \(r, r'\) 가 각각 서로 다른 정책 \(\pi,\pi'\) 를 써서

\[
r(x,y)=\beta \log \frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)},\qquad
r'(x,y)=\beta \log \frac{\pi'(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\]

로 표현된다고 가정해 보자. 같은 동치류이므로 \(r'(x,y)=r(x,y)+f(x)\) 여야 한다. 두 식을 결합하면 \(\pi'(y\mid x)=\pi(y\mid x)\exp(f(x)/\beta)\) 가 된다. 하지만 \(\pi\) 와 \(\pi'\) 는 둘 다 확률분포이므로 \(y\) 에 대해 합하면 \(\exp(f(x)/\beta)=1\), 즉 \(f(x)=0\) 이어야 한다. 따라서 결국 \(r=r'\) 다. 즉 DPO 재매개변수화는 동치류마다 딱 하나의 대표 보상을 고른다.

이 정리와 명제의 결론을 한 줄로 요약하면 이렇다.  
**DPO는 보상 동치류의 표현력을 버리지 않으면서, 각 동치류를 정책으로 직접 표시할 수 있는 정규화된 대표 보상 함수로 바꾼다.**

## B. DPO Implementation Details and Hyperparameters

부록 B는 DPO 구현이 실제로 얼마나 짧은지 보여 준다. 논문은 PyTorch 코드 몇 줄로 손실을 계산할 수 있다고 강조한다. 핵심 계산은 다음 단계로 요약된다.

1. 정책 모델과 기준 모델이 각 completion에 부여한 로그확률을 준비한다.  
2. 선호된 응답 \(y_w\) 와 비선호 응답 \(y_l\) 의 인덱스로 두 모델의 로그확률을 각각 뽑는다.  
3. 정책 로그비 \( \log \pi_\theta(y_w\mid x)-\log \pi_\theta(y_l\mid x) \) 와 기준 로그비 \( \log \pi_{\mathrm{ref}}(y_w\mid x)-\log \pi_{\mathrm{ref}}(y_l\mid x) \) 를 계산한다.  
4. 두 로그비의 차이에 \(\beta\) 를 곱하고 \(-\log \sigma(\cdot)\) 를 취하면 손실이 된다.  
5. 동시에
   \[
   \beta(\log \pi_\theta(y\mid x)-\log \pi_{\mathrm{ref}}(y\mid x))
   \]
   를 detach하여 암묵적 보상처럼 모니터링할 수 있다.

이를 의사코드 형태로 적으면 아래와 같다.

```python
pi_yw_logps  = pi_logps[yw_idxs]
pi_yl_logps  = pi_logps[yl_idxs]
ref_yw_logps = ref_logps[yw_idxs]
ref_yl_logps = ref_logps[yl_idxs]

pi_logratios  = pi_yw_logps - pi_yl_logps
ref_logratios = ref_yw_logps - ref_yl_logps

losses  = -logsigmoid(beta * (pi_logratios - ref_logratios))
rewards = beta * (pi_logps - ref_logps).detach()
```

손실 구현이 짧은 이유는 DPO가 보상 모델 학습 단계와 RL 단계 사이를 오가는 복잡한 상호작용을 거의 제거했기 때문이다. 정책 네트워크 하나와 고정 reference만 있으면 된다. 논문이 기본값으로 사용하는 설정은 다음과 같다.

- 기본 \(\beta = 0.1\)
- 배치 크기 64
- RMSprop 옵티마이저
- 학습률 \(1\times10^{-6}\)
- 처음 150 step 동안 선형 warmup
- TL;DR 요약에서는 \(\beta = 0.5\)

이 설정은 놀라울 정도로 단순하다. 논문이 DPO를 실용적인 방법으로 밀어붙이는 이유도 여기에 있다. 하이퍼파라미터의 종류 자체가 PPO 기반 RLHF보다 적고, value model이나 reward normalization을 따로 붙일 필요도 없다.

## C. Further Details on the Experimental Set-Up

부록 C는 본문 실험을 재현하고 해석하는 데 필요한 세부 사항을 모은 부분이다. 통제된 IMDb 감성 생성에서 데이터가 어떻게 만들어졌는지, GPT-4를 어떻게 비교 평가자로 썼는지, 그리고 왜 unlikelihood 기준선이 일부 과제에서 제외되었는지가 정리되어 있다.

### C.1 IMDb Sentiment Experiment and Baseline Details

IMDb 감성 제어 실험은 DPO와 PPO의 reward-KL frontier를 가장 직접적으로 비교하기 위한 통제 실험이다. 프롬프트는 IMDb 데이터셋에서 길이 2~8 토큰인 리뷰 접두부를 사용한다. 실제 보상 함수는 `siebert/sentiment-roberta-large-english` 감성 분류기로 두고, 기본 언어 모델은 `gpt2-large` 를 사용한다. 저자들은 더 작은 기본 모델이나 더 약한 감성 모델은 텍스트 품질이 떨어지거나 보상 판정이 불안정해져 이 비교 실험에 적합하지 않았다고 설명한다.

절차는 다음과 같다. 먼저 IMDb 일부 데이터로 1 epoch 동안 SFT를 수행한다. 그다음 이 모델로 25,000개의 접두부에 대해 각 4개의 completion을 샘플링하고, 감성 분류기 점수에 따라 각 접두부마다 6개의 선호쌍을 만든다. RLHF reward model은 `gpt2-large` 에서 초기화해 3 epoch 학습하며, 검증 정확도가 가장 높은 체크포인트를 사용한다. PPO 비교에서는 TRL 라이브러리 기본 설정으로 돌린 버전과, 보상 정규화와 추가 튜닝을 넣은 저자 구현 버전을 모두 사용한다. 저자 구현 PPO는 PPO step마다 더 큰 배치 샘플 1024를 사용한다.

이 세부 설정이 중요한 이유는 본문 Figure 2 왼쪽의 frontier가 단순히 "대충 비슷한 설정에서 누가 좋아 보이는가"가 아니라, 가능한 한 공정하고 통제된 조건에서 나온 결과임을 보여 주기 때문이다.

### C.2 GPT-4 prompts for computing summarization and dialogue win rates

논문은 GPT-4 승률 계산에 사용한 프롬프트 구조도 공개한다. 모든 실험에는 `gpt-4-0314` 를 사용하며, 비교되는 두 요약 또는 두 응답의 순서는 매 평가마다 무작위로 섞는다.

요약용 프롬프트는 두 가지다.

첫 번째 **GPT-4 (S)** 는 "주어진 포럼 글의 가장 중요한 포인트를 더 잘 요약한 것이 어느 쪽인가"를 묻는다. GPT-4는 먼저 한 문장 비교 설명을 쓰고, 다음 줄에 A 또는 B만 출력해야 한다.

두 번째 **GPT-4 (C)** 는 여기에 조건을 하나 더 더한다. 단순히 중요한 내용을 담는지뿐 아니라, **불필요하거나 무관한 세부를 넣지 않는지**, 즉 정확하면서도 간결한지까지 평가한다. 본문 6.4절에서 보듯이 이 프롬프트가 인간 취향에 더 가까운 결과를 준다.

대화용 프롬프트는 "다음 사용자 질의에 대해 어느 응답이 더 도움이 되는가"를 묻는다. 이때도 GPT-4는 먼저 한 문장 설명을 출력하고, 다음 줄에 A 또는 B를 적는다. 핵심은 이 모든 프롬프트가 **점수화** 가 아니라 **쌍대 비교** 를 수행한다는 점이다. 따라서 DPO가 학습한 선호 구조와 평가 방식의 형태가 잘 맞아떨어진다.

### C.3 Unlikelihood baseline

Unlikelihood 기준선은 선호된 응답 \(y_w\) 의 로그확률은 높이고 비선호 응답 \(y_l\) 의 로그확률은 낮추는 단순한 목적이다. 감성 제어 실험에서는 비교를 위해 포함했지만, 논문은 요약과 대화에서는 이 방법을 제외한다. 이유는 간단하다. 복잡한 생성 과제에서는 이 목적이 모델을 **의미 있는 응답 생성** 이 아니라 **무작정 비선호 응답을 피하는 방향** 으로 몰아, 결과적으로 품질이 붕괴하기 때문이다.

[Table 3 원문 표 및 원문 캡션 삽입]

Table 3의 예시는 그 붕괴를 아주 직관적으로 보여 준다. TL;DR 요약 프롬프트에 대해 unlikelihood 모델은 정상적인 요약 대신 반복적인 토큰 나열과 무의미한 문자열을 출력한다. 즉, 선호된 응답을 높이고 비선호 응답을 낮추는 것만으로는 개방형 생성에서 좋은 출력을 보장할 수 없다. DPO가 단순한 확률 차이 이상의 구조를 가지는 이유가 바로 여기에 있다. reference 대비 로그비, 그리고 예제별 가중이 없는 목적은 언어 모델을 쉽게 퇴화시킨다.

## D. Additional Empirical Results

부록 D는 본문에서 요약된 결과를 더 구체적인 예시와 추가 그래프로 보여 준다. Best of N이 실제로 어디서 포화되는지, GPT-4가 DPO와 기준선을 비교할 때 어떤 응답을 더 선호했는지, 인간 평가 실험이 어떤 방식으로 진행되었는지를 확인할 수 있다.

### D.1 Performance of Best of N baseline for Various N

Best of N은 계산량이 크지만 강한 기준선이다. 그렇다면 \(N\) 이 커질수록 성능이 얼마나 계속 올라갈까? 저자들은 Anthropic HH 대화와 TL;DR 요약 두 과제에서 \(N\) 을 바꿔가며 이 곡선을 그린다.

[Figure 4 원문 이미지 및 원문 캡션 삽입]

Figure 4의 메시지는 명확하다. 두 과제 모두 Best of N 성능은 대략 **64~128개 샘플 부근에서 포화** 된다. 즉, 샘플을 무한히 많이 늘린다고 끝없이 좋아지는 것이 아니다. 이 결과는 본문에서 DPO를 Best of 128과 비교하는 것이 어느 정도 타당한 기준선이 된다는 점을 뒷받침한다. 특히 대화 과제에서는 DPO가 이 매우 비싼 Best of 128과 맞먹는 수준까지 올라가므로, DPO가 단순한 계산 절감형 근사치가 아니라 상당히 경쟁력 있는 정책 학습법이라는 점이 더 분명해진다.

### D.2 Sample Responses and GPT-4 Judgments

이 절은 DPO가 실제 샘플 수준에서 어떤 차이를 만드는지 보여 준다. Tables 4~6은 TL;DR 요약 사례, Tables 7~10은 Anthropic HH 대화 사례다. 요약 사례에서는 DPO와 PPO를 비교하고, 대화 사례에서는 DPO와 데이터셋의 chosen 응답을 비교한다. 논문은 실제 GPT-4 평가 시 두 응답의 순서를 무작위로 섞었고, 표에 붙은 대괄호 주석은 이해를 돕기 위한 사후 표기이지 모델 출력 자체는 아니라고 밝힌다.

[Table 4 원문 표 및 원문 캡션 삽입]

Table 4는 "부모님께 못되게 굴어 온 것을 후회하는 17세 사용자가, 대학에 가기 전과 부모님 생일·크리스마스에 무엇을 하면 좋을지 묻는 글"을 요약한 예시다. DPO 요약은 문제의 핵심을 "부모님께 사랑과 감사의 마음을 보여 주기 위해 어떤 행동을 해야 하는가"로 압축한다. 반면 PPO 요약은 같은 질문을 여러 형태로 반복하면서 문장을 늘여 쓴다. GPT-4는 DPO 쪽이 핵심을 더 분명하게 잡고 있고 반복이 덜하다고 판단한다. 이 사례는 DPO가 단순히 더 짧은 요약을 만드는 것이 아니라, **핵심 질문을 중심으로 압축하는 능력** 이 있음을 보여 준다.

[Table 5 원문 표 및 원문 캡션 삽입]

Table 5에서는 "오랜 친구이자 거의 2년 사귄 여자친구에게 곧 청혼할 예정인데 너무 긴장돼 잠이 오지 않는다"는 관계 고민 글이 나온다. DPO 요약은 "청혼을 앞두고 극심한 불안으로 잠도 못 자는데 어떻게 진정할 수 있을까"라는 한 문장으로 깔끔하게 정리된다. PPO 요약은 청혼 사실과 진정 방법, 수면 문제를 담고는 있지만 줄바꿈과 반복이 많아 요약문 자체가 흐트러져 있다. GPT-4는 DPO가 질문자의 핵심 상태와 요청을 더 간결하고 선명하게 전달한다고 본다.

[Table 6 원문 표 및 원문 캡션 삽입]

Table 6은 저칼로리 파스타 대체재를 찾는 글이다. DPO 요약은 "칼로리를 줄이기 위해 일반 파스타와 비슷한 고섬유 파스타를 찾았는데 정말 괜찮은 선택인지 묻는다"는 식으로 원문의 의도를 보수적으로 요약한다. PPO는 이를 체중 감량 문제로 더 강하게 해석해 "살을 빼고 싶다"는 의도를 덧붙인다. GPT-4는 DPO가 **원문에 없는 동기를 억지로 덧붙이지 않고**, 실제 질문 의도인 "이 제품이 과장인지 아닌지"에 더 충실하다고 판단한다. 이 예시는 DPO가 요약에서 부정확한 추론을 덜 얹는 경향이 있음을 보여 준다.

[Table 7 원문 표 및 원문 캡션 삽입]

Table 7은 "미국 시민권 운동에 대한 에세이를 쓰는 데 도움을 줄 수 있는가"라는 질문이다. DPO는 역사적 배경과 가능한 에세이 방향을 길게 제안하면서 사용자가 어떤 종류의 에세이를 원하는지 되묻는다. 데이터셋의 chosen 응답은 훨씬 짧고, 실제 글을 대신 써 주기보다는 어떤 점을 다루고 싶은지부터 묻는다. GPT-4는 DPO 응답이 더 구체적인 정보와 다음 단계 제안을 함께 제공하므로 더 도움이 된다고 판단한다. 이 예시는 DPO가 단순히 더 친절한 톤을 만드는 것이 아니라, **문제 해결에 필요한 정보 밀도** 를 높일 수 있음을 보여 준다.

[Table 8 원문 표 및 원문 캡션 삽입]

Table 8은 개인 주소를 찾아 달라는 프라이버시 침해 요청이다. DPO는 그런 정보 제공을 도울 수 없다고 명확히 거절하고, 동의 없는 개인정보 수집은 불법적이고 무책임하다고 말한다. 반면 chosen 응답은 직접 주소를 주지는 않지만 대상 인물의 배경 정보와 재산 관련 사실을 늘어놓는다. GPT-4는 DPO가 사생활 보호 원칙을 더 분명히 지키고 있다고 판단한다. 이 사례는 DPO가 "도움됨"만이 아니라 **무해성과 경계 설정** 에서도 stronger alignment를 만들 수 있음을 보여 준다.

[Table 9 원문 표 및 원문 캡션 삽입]

Table 9는 "미국이 제2차 세계대전에 개입하게 된 계기가 무엇인가"라는 역사 질문이다. 여기서는 DPO가 오히려 패한다. DPO 응답은 길고 그럴듯하게 들리지만, 대공황과 국제 질서, 'coalition of the willing' 같은 역사적으로 맞지 않는 내용을 섞어 잘못된 설명을 한다. chosen 응답은 훨씬 직설적으로 진주만 공습과 그 뒤의 대일 선전포고를 말한다. GPT-4는 당연히 chosen 쪽을 더 낫다고 판단한다. 이 예시는 DPO가 항상 이기는 것이 아니며, 특히 **그럴듯한 장광설이 사실 오류를 감출 수 있다** 는 한계를 보여 준다.

[Table 10 원문 표 및 원문 캡션 삽입]

Table 10은 아주 짧은 산수 질문 "7+2는 얼마인가"에 대한 사례다. DPO 응답은 처음에는 9라고 맞게 말하는 듯하지만, 곧 불필요하게 장황한 설명과 이상한 "arithmetic-prefix method" 같은 말을 늘어놓는다. chosen 응답은 단순히 11을 제시해 틀린 답을 준다. 흥미롭게도 GPT-4는 여기서 chosen 쪽을 더 직접적이고 정확하다고 잘못 판단한다. 이 사례는 두 가지를 보여 준다. 하나는 DPO가 간단한 질문에서도 불필요하게 verbose해질 수 있다는 점이고, 다른 하나는 GPT-4 평가 역시 완벽하지 않다는 점이다. 논문이 6.4절과 D.3절에서 인간 검증을 추가한 이유가 정확히 여기에 있다.

### D.3 Human study details

[Figure 5 원문 이미지 및 원문 캡션 삽입]

이 절은 GPT-4 평가의 신뢰성을 검증하기 위해 수행한 인간 평가 실험의 세부 설계를 설명한다. 비교 조합은 세 가지다. DPO(temperature 0.25) 대 PPO(temperature 0), SFT(temperature 0.25) 대 PPO(temperature 0), PPO(temperature 1.0) 대 PPO(temperature 0)다. 성능이 좋은 경우, 중간인 경우, 나쁜 경우를 모두 포함해 GPT-4와 인간의 승률 관계가 응답 품질 스펙트럼 전반에서 어떻게 움직이는지 보려는 설계다.

샘플 수는 DPO 대 PPO-0 비교 150개, PPO-1 대 PPO-0 비교 100개, SFT 대 PPO-0 비교 125개다. DPO 대 PPO와 PPO-1 대 PPO 비교에는 각 비교당 두 명의 인간을 배정했고, SFT 비교에는 한 명을 배정했다. 인간이 tie를 고른 경우는 약 1%뿐이라 대부분의 비교는 선호 방향이 분명했다. 논문은 인간 A와 인간 B의 raw agreement, 그리고 각 인간과 GPT-4 사이의 raw agreement를 함께 측정한다.

참가자는 총 25명의 자원봉사자였다. 주로 스탠퍼드 학부생, 대학원생, 최근 졸업생, 방문연구원으로 구성되었고 STEM, 특히 컴퓨터과학 배경이 많았다. 논문은 SurveyMonkey 인터페이스 화면도 Figure 5로 제시한다. 참가자 목록은 다음과 같다.

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

논문은 한 명의 자원봉사자가 DPO 대 PPO 비교 응답을 끝내지 못했다고 덧붙인다. 이 부록 전체의 결론은 단순하다. GPT-4 평가는 충분히 유용하지만, 완벽한 대체물이 아니므로 인간 검증을 함께 보는 것이 가장 안전하다.

## 추가 설명

논문 본문을 다 따라온 뒤 마지막에 남는 질문은 결국 이것이다. **DPO는 왜 이렇게 단순한데도 RLHF와 비슷하거나 더 잘 되는가?** 이 절에서는 그 직관을 한 번 더 정리한다.

### 1. DPO는 RLHF의 목적을 버린 것이 아니라, 해를 직접 쓴 것이다

많은 설명이 DPO를 "RL을 없앤 간단한 근사"처럼 소개하지만, 이 논문의 핵심은 그것이 아니다. DPO는 본문 식 (3)의 KL 제약 보상 최대화 목적을 다른 것으로 바꾸지 않는다. 오히려 그 목적의 최적 정책이 식 (4)처럼 닫힌형태로 쓰인다는 사실을 이용해, 보상 함수를 정책의 함수로 직접 바꿔 버린다. 따라서 DPO는 RLHF를 대체하는 임시방편이 아니라, **RLHF 문제의 재매개변수화** 로 이해하는 것이 정확하다.

### 2. 기준 정책 \( \pi_{\mathrm{ref}} \) 는 단순한 초기값이 아니라 비교의 좌표계다

DPO 손실에는 항상
\[
\log \frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\]
가 등장한다. 이는 현재 정책이 특정 응답을 얼마나 좋아하는지를 절대적으로 재는 것이 아니라, **기준 정책에 비해 얼마나 더 밀어주었는가** 를 재는 것이다. 그래서 reference는 단순한 초기 체크포인트 이상의 의미를 가진다. reference가 바뀌면 같은 정책도 다른 암묵적 보상으로 해석된다. DPO가 공개 선호 데이터셋을 사용할 때 원래 SFT 모델을 reference로 두려 하는 이유가 여기에 있다.

### 3. \(\beta\) 는 온도가 아니라 보수성의 손잡이다

식 (4)와 식 (7)에 동시에 등장하는 \(\beta\) 는 DPO의 가장 중요한 하이퍼파라미터다. \(\beta\) 가 작으면 보상 차이가 더 크게 증폭되어 reference로부터 더 과감히 이동하고, \(\beta\) 가 크면 이동이 보수적이 된다. DPO가 사실상 KL 제약을 손실 안에 흡수하고 있기 때문에, \(\beta\) 는 "선호를 얼마나 세게 따를 것인가"와 "reference를 얼마나 존중할 것인가" 사이의 균형을 직접 조절한다. 본문 실험에서 \(\beta\) 를 많이 튜닝하지 않았는데도 결과가 좋았다는 점은 DPO의 실용성을 보여 준다.

### 4. DPO의 가중 항은 단순한 디테일이 아니라 품질 유지 장치다

DPO를 아주 단순하게 흉내 내면 "chosen 확률 올리고 rejected 확률 내리기"가 되지만, 논문은 그 방식이 생성 모델을 쉽게 망가뜨린다고 본다. 핵심 차이는 기울기에 들어가는
\[
\sigma(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w))
\]
가중이다. 이 항은 아직 잘못 정렬된 예제에 더 큰 업데이트를 주고, 이미 잘 정렬된 예제는 과도하게 더 밀지 않게 한다. 즉 DPO는 모든 preference pair를 동일 강도로 압박하지 않는다. 부록 Table 3의 unlikelihood 붕괴 예시는 왜 이런 가중이 필요한지 보여 주는 실전 사례다.

### 5. PPO보다 잘 되는 이유는 "목적이 달라서"가 아니라 "최적화 경로가 짧아서"일 수 있다

논문은 DPO와 PPO가 같은 목적을 푼다고 본다. 그런데도 Figure 2에서 DPO가 더 나은 frontier를 보이는 이유는, PPO가 보상 모델, value function, reward normalization, sampling noise, on-policy update라는 여러 근사와 불안정성을 거쳐야 하기 때문이다. 반면 DPO는 오프라인 선호쌍을 바로 BCE 형태로 최적화한다. 따라서 성능 차이는 "누가 더 좋은 objective를 가졌는가"보다 **누가 더 적은 근사와 적은 분산으로 그 objective를 최적화하는가** 의 차이로 읽는 편이 자연스럽다.

### 6. DPO의 한계는 여전히 남아 있다

논문이 보여 주는 것은 DPO가 강력하다는 사실이지, 만능이라는 사실은 아니다. Table 9와 Table 10이 보여 주듯 DPO는 그럴듯하지만 틀린 장문 응답을 만들 수 있고, 간단한 문제에서도 지나치게 장황해질 수 있다. 또한 GPT-4 평가 자체도 오판한다. 결국 DPO는 학습 파이프라인을 단순화했을 뿐, 사실성·간결성·OOD 일반화 같은 큰 문제를 자동으로 해결하지는 않는다. 그렇기 때문에 저자들이 마지막 Discussion에서 더 큰 모델, 더 넓은 분포 이동, self-labeling, reward over-optimization을 후속 과제로 남긴 것이다.

논문 전체를 한 문장으로 요약하면 이렇다.  
**DPO는 인간 선호 학습을 RL 문제로 우회하지 않고, 정책이 이미 보상 모델 역할을 할 수 있다는 사실을 이용해 선호 최적화를 직접 수행하는 방법이다.**
