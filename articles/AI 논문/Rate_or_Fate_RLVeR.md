
# Rate or Fate? RLVεR: Reinforcement Learning with Verifiable Noisy Rewards

## Abstract

RLVR(Reinforcement Learning with Verifiable Rewards)는 한 프롬프트에 대해 여러 개의 completion을 생성하고, 각 completion이 정답인지 자동으로 검증한 다음, 그 결과를 이용해 정책을 업데이트하는 학습 패러다임이다. 수학과 코드처럼 정답 여부를 비교적 명확하게 판단할 수 있는 영역에서는 이 방식이 매우 강력하다. 그러나 논문이 문제 삼는 지점은 바로 그 “검증 가능성”이 실제로는 결코 완전하지 않다는 데 있다. 코드의 unit test는 모든 반례를 검사하지 못하고, 사람 혹은 합성 데이터에서 만든 라벨은 오차를 포함하며, LLM judge 역시 편향되거나 exploit될 수 있다. 따라서 RLVR의 진짜 핵심 문제는 “보상이 있느냐”가 아니라 “그 보상이 얼마나 믿을 만하냐”이다.

이 논문은 이 문제를 아주 직접적으로 묻는다. 검증기의 노이즈는 단지 학습 속도만 늦추는가, 아니면 학습의 최종 방향 자체를 바꾸는가. 이를 답하기 위해 저자들은 RLVR를 해석 가능한 multi-armed bandit으로 추상화하고, completion들을 반복적으로 나타나는 reasoning mode의 집합으로 묶는다. 그러면 정책은 거대한 시퀀스 공간 위의 분포가 아니라, 소수의 good mode와 bad mode 위에 놓인 확률벡터로 바뀐다. 이 표현 아래에서 GRPO 업데이트는 확률 단체(simplex) 위의 replicator flow, 즉 자연선택형 동역학으로 다시 읽힌다.

논문이 도출하는 핵심 변수는 잘못된 모드에 실린 총 확률 질량 \(p\)와, 검증기의 순 판별력을 나타내는 Youden 지수 \(J\)이다. 정답과 오답을 각각 noisy verifier가 뒤집을 수 있다고 두면, 거짓 음성률과 거짓 양성률은 다음처럼 정의된다.

$$
\delta_{\mathrm{FN}}=\Pr(r=0\mid \text{good}),\qquad
\delta_{\mathrm{FP}}=\Pr(r=1\mid \text{bad})
\tag{1}
$$

그리고 두 값을 하나의 스칼라로 요약한 것이

$$
J=\mathrm{TPR}-\mathrm{FPR}=(1-\delta_{\mathrm{FN}})-\delta_{\mathrm{FP}}
\tag{2}
$$

이다. 이 값이 양수이면 verifier는 최소한 random guessing보다 낫고, 0이면 전혀 정보가 없으며, 음수이면 아예 잘못된 방향의 신호를 준다.

논문의 결론은 놀랄 만큼 날카롭다. GRPO형 group-normalized RLVR에서는 bad mass \(p\)의 drift 부호가 오직 \(J\)의 부호로만 결정된다. \(J>0\)이면 bad mass가 줄어들면서 학습이 일어나고, \(J=0\)이면 drift가 사라져 중립 표류가 되며, \(J<0\)이면 bad mass가 오히려 증폭되어 anti-learning이 발생한다. 즉, noisy reward는 항상 “rate의 문제”인 것이 아니라, 임계값을 넘는 순간 “fate의 문제”가 된다.

또한 이 논문은 learning regime, 즉 \(J>0\)에서는 noisy reward와 clean reward가 장기적으로 같은 basin으로 향한다는 점을 보인다. 노이즈는 대체로 시간축을 늘려 학습을 더 느리게 만들 뿐, 최종 도착점 자체를 바꾸지 않는다. 이것이 논문 제목의 “Rate or Fate?”에 대한 대답이다. 다만 \(J\le 0\)가 되면 이야기 자체가 바뀐다. 이때는 계산을 더 쓰는 것이 도움이 되지 않고, 오히려 잘못된 모드를 더욱 강화할 수 있다.

실험은 Python 코드 생성 과제에서 합성 verifier noise를 주입해 이 예측을 검증한다. 결과는 이론과 같은 질적 구조를 보인다. \(J>0\)에서는 pass@1이 개선되고, \(J=0\)에서는 거의 변화가 없으며, \(J<0\)에서는 성능이 실제로 악화된다. 더 나아가 같은 \(J\)라도 FP와 FN의 배치에 따라 속도와 성능이 달라질 수 있음을 보여 주어, noisy RLVR를 설계할 때 단순 평균 정확도 이상의 관점이 필요하다는 점도 드러낸다.

<figure>
  <img src="/assets/images/Rate_or_Fate_RLVeR/rlver_figure_01_phase_transition_under_a_sloppy_oracle.png" alt="Figure 1. Phase transition under a sloppy oracle." />
  <figcaption><strong>Figure 1.</strong> Phase transition under a sloppy oracle.</figcaption>
</figure>

## 1 Introduction

### Reinforcement Learning and LLMs.

서론의 출발점은 RL, 특히 RLVR와 GRPO류의 group-normalized 알고리즘이 최근 LLM의 수학·코드 추론 능력을 눈에 띄게 끌어올렸다는 사실이다. RLHF처럼 별도의 reward model을 학습시키는 대신, RLVR은 정답 여부를 직접 검증할 수 있는 과제에서 sequence-level reward를 바로 사용한다. 이는 보상을 모델이 따로 예측하지 않아도 된다는 장점이 있고, 몇 개의 rollout만으로도 completion 간 상대적 우열을 비교해 학습 방향을 만들 수 있다는 점에서 계산적으로도 매력적이다.

하지만 이 구조의 중심은 여전히 보상이다. critic를 없앴다고 해서 reward의 질 문제가 없어진 것이 아니며, 오히려 verifier가 곧 rewarder가 되는 만큼 그 verifier가 얼마나 정확한지가 학습 전체를 좌우한다. 논문은 바로 이 지점을 파고든다. RLVR의 성공담 뒤에는 보상이 충분히 깨끗하다는 암묵적 가정이 숨어 있고, 실제 환경에서는 그 가정이 자주 깨진다.

### Group-Normalized RL and RLVR.

GRPO의 기본 반복은 단순하다. 한 프롬프트 \(x\)에 대해 현재 정책 \(\pi_\theta\)로 여러 개의 completion \(y_1,\dots,y_G\)를 샘플링하고, 각 completion에 raw reward \(r_i\)를 붙인다. 그런 다음 그룹 내부 평균과 표준편차를 이용해 reward를 표준화하여 advantage를 만든다.

$$
A_i=\frac{r_i-\bar r}{s_r}
$$

여기서 \(\bar r\)는 그룹 평균 reward이고, \(s_r\)는 그룹 표준편차다. 이후 정책은 높은 advantage를 받은 completion의 로그확률을 올리고, 낮은 advantage를 받은 completion의 로그확률을 내리는 방향으로 갱신된다.

$$
\Delta \theta \propto \sum_{i=1}^{G} A_i \nabla_\theta \log \pi_\theta(y_i\mid x)
\tag{3}
$$

이 식의 의미는 분명하다. GRPO는 “절대적으로 좋은 답”을 학습하는 것이 아니라, 같은 프롬프트에서 생성된 답들 사이의 상대적 우열을 학습한다. 그러므로 verifier가 주는 reward의 절대값보다, good completion과 bad completion을 얼마나 일관되게 구분하느냐가 중요해진다.

### Noisy Reward.

논문은 정답 여부를 나타내는 잠재적 참값과, verifier가 실제로 돌려주는 operational reward를 구분한다. 참값이 깨끗하더라도 verifier가 그 참값을 뒤집을 수 있기 때문이다. 이때 핵심 오류는 두 가지뿐이다. 정답을 틀렸다고 하는 false negative, 오답을 맞았다고 하는 false positive다. 논문은 이 둘을 식 (1)로 정의하고, 다시 식 (2)의 Youden 지수 \(J\)로 요약한다.

\(J\)는 통계적 의미가 분명한 양이다. \(J=1\)이면 완벽한 verifier다. \(J=0\)이면 ROC 직선 위의 random guess와 같아서, 보상 신호가 장기적으로 아무 방향도 주지 못한다. \(J<0\)이면 verifier가 systemically inverted되어, 오답을 정답보다 더 우대하는 셈이 된다. 논문은 noisy RLVR를 해석할 때 수많은 세부 파라미터를 모두 쫓기보다, 이 \(J\) 하나가 학습 방향을 결정하는 중심 변수라고 주장한다.

### Noisy Reward for Coding Tasks.

저자들이 실험 무대로 코드 생성을 고른 이유도 여기 있다. 수학처럼 답이 하나의 문자열로 정리되는 문제와 달리, 코드는 동일한 기능을 구현하는 서로 다른 정답이 많다. 반면 unit test는 유한한 입력 집합만 검사하므로, 잘못된 프로그램이 테스트를 통과하는 false positive도 생길 수 있고, 올바른 프로그램이 테스트 형식이나 edge case 때문에 실패 처리되는 false negative도 생길 수 있다. 코딩 RLVR는 noisy verifier 문제가 실제로 가장 자연스럽게 드러나는 환경이다.

논문이 이 서론에서 제기하는 질문은 네 가지다. 첫째, RLVR는 어느 정도의 reward sloppiness까지 견딜 수 있는가. 둘째, noisy reward 아래에서 학습 동역학은 구체적으로 어떻게 생기는가. 셋째, 학습이 갑자기 실패하거나 방향이 뒤집히는 임계점이 존재하는가. 넷째, 단순히 성능 곡선을 관찰하는 것이 아니라, accuracy와 직접 연결되는 상태변수의 drift를 닫힌형태 ODE로 써서 rate와 fate를 분리할 수 있는가. 이후 본문은 이 네 질문을 거의 그대로 따라간다.

강화학습은 본질적으로 feedback-driven 시스템이다. 그래서 ground truth가 완전히 없어도 학습이 가능하냐는 질문은, 사실상 noisy feedback만으로 self-improvement가 가능한지 묻는 것과 같다. 논문은 이 문제를 추상적 직관이 아니라 해석 가능한 mean-field 동역학으로 바꾸어 논의한다는 점에서 의미가 있다.

<figure>
  <img src="/assets/images/Rate_or_Fate_RLVeR/rlver_figure_02_the_winner_takes_it_all.png" alt="Figure 2. The winner takes it all." />
  <figcaption><strong>Figure 2.</strong> The winner takes it all.</figcaption>
</figure>

Figure 2가 서론에서 미리 보여 주는 것은 GRPO류 동역학의 구조적 성질이다. 학습 regime에서는 bad mass가 0으로 가는 동시에, good arm들 사이에서는 초기 질량이 가장 컸던 arm이 결국 승자가 된다. 경계 \(J=0\)에서는 확률벡터 전체가 정지하고, \(J<0\)에서는 bad arm이 승자가 된다. 즉, noisy reward 문제는 단지 “좋은 답이 얼마나 늘어나느냐”의 문제가 아니라, 모드 간 확률 질량이 어떻게 재배치되느냐의 문제다.

## 2 RLVR with Sloppy Rewards

RLVR의 핵심 루프는 샘플링, 채점, 그룹 정규화, 정책 업데이트의 네 단계로 이루어진다. 샘플링은 현재 정책이 어떤 답을 얼마나 자주 내는지를 측정하는 과정이고, 채점은 각 답에 verifier가 붙이는 raw reward를 얻는 과정이다. 그룹 정규화는 이 raw reward를 같은 프롬프트 내부의 상대 비교값으로 바꾸는 과정이며, 마지막 정책 업데이트는 그 상대 비교를 log-probability gradient로 정책에 반영하는 단계다.

논문이 여기서 강조하는 점은, GRPO가 “reward를 얼마나 많이 받았는가”보다 “같은 그룹 내 평균보다 얼마나 낫거나 못한가”를 학습한다는 것이다. 그래서 노이즈는 raw reward의 절대값을 흐리는 문제가 아니라, good completion과 bad completion의 상대적 우위를 뒤집을 수 있다는 점에서 더 치명적이다. GRPO는 noisy reward를 평균내어 없애는 방식이 아니라, 그 noisy reward가 만들어 낸 상대적 우위 구조를 오히려 증폭한다.

$$
A_i=\frac{r_i-\bar r}{s_r},\qquad
\Delta \theta \propto \sum_{i=1}^{G} A_i \nabla_\theta \log \pi_\theta(y_i\mid x)
\tag{3 revisited}
$$

이 정규화 때문에 reward의 스케일 자체는 중요하지 않다. 중요한 것은 good과 bad 사이의 normalized gap이 양수인지 음수인지다. 이후 전개는 결국 이 gap을 해석하고, 그 gap의 부호가 무엇으로 결정되는지를 밝히는 방향으로 진행된다.

<figure>
  <img src="/assets/images/Rate_or_Fate_RLVeR/rlver_figure_03_rate_not_fate.png" alt="Figure 3. Rate, not fate." />
  <figcaption><strong>Figure 3.</strong> Rate, not fate.</figcaption>
</figure>

Figure 3이 미리 보여 주는 메시지는 간단하다. 학습이 가능한 구간에서는 noisy reward가 최종 정확도를 본질적으로 바꾸지 않고, 동일한 궤적을 더 느리게 따라가게 만든다. 즉, 노이즈는 경로를 흐리게 만들지만 basin을 바꾸지는 않는다. 다만 이 결론은 어디까지나 \(J>0\)에서만 성립한다.

### 2.1 Learning Dynamics

#### Good vs. Bad.

논문은 먼저 가장 단순한 binary setup을 본다. 모델이 한 프롬프트에 대해 good solution 아니면 bad solution 둘 중 하나만 낸다고 가정하고, bad solution이 나올 확률을 \(p\)라고 둔다. 이때 \(p\)는 어떤 로짓 \(\beta\)의 sigmoid로 생각할 수 있다.

정책 업데이트는 normalized advantage와 score function의 상관으로 정해진다. bad와 good 각각에 대한 조건부 평균 advantage를 \(\bar A_{\text{bad}}, \bar A_{\text{good}}\)라 두면, 로짓 \(\beta\)의 drift는 두 평균 advantage 차이로 요약된다. 그리고 로짓에서 확률 \(p\)로 pushforward하면 bad mass 자체의 법칙이 나온다.

$$
\dot p
=
-\eta [p(1-p)]^2\bigl(\bar A_{\text{good}}-\bar A_{\text{bad}}\bigr)
\tag{4}
$$

이 식이 중요한 이유는 \([p(1-p)]^2\)라는 prefactor 때문이다. bad mass가 거의 0이거나 거의 1이면, 아무리 advantage gap이 남아 있어도 업데이트량 자체는 작아진다. 즉, 모델이 이미 거의 다 맞히거나 거의 다 틀리는 프롬프트는 학습이 느리다. 반대로 중간 영역에서는 같은 gap이라도 훨씬 큰 업데이트가 생긴다.

여기서 제곱이 붙는 이유도 해석적으로 명확하다. 하나의 \(p(1-p)\)는 score function 기대값에서 나오고, 또 하나의 \(p(1-p)\)는 로짓 동역학을 확률 동역학으로 옮길 때 sigmoid 미분에서 나온다. 이 때문에 GRPO의 two-state dynamics는 단순한 선형 drift가 아니라, 확률 경계에서 자연스럽게 멈추는 multiplicative system이 된다.

#### Interpretation.

식 (4)는 전형적인 replicator dynamics의 골격을 가진다. 평균보다 fitness가 높은 타입은 비중이 늘고, 평균보다 낮은 타입은 비중이 줄어든다. 여기서 fitness에 해당하는 것이 normalized advantage다. good 답이 평균보다 일관되게 높은 advantage를 받으면 good 질량이 커지고, bad 답이 높으면 bad 질량이 커진다.

이 관점에서 보면 GRPO는 “자연선택 같은 알고리즘”이다. 평균보다 나은 completion은 살아남고, 평균보다 못한 completion은 사라진다. 논문은 이 해석을 Section 5에서 확률 단체 위의 자연기하와 연결하고, Appendix I와 J에서 mode 내부 경쟁까지 추적한다.

### 2.2 Noisy Rewards

#### Signal vs. Noise.

이제 verifier가 clean하지 않다고 하자. bad mass \(p\)가 주어졌을 때, 한 번의 샘플이 받을 noisy reward의 평균은

$$
m(p)
=
(1-p)(1-\delta_{\mathrm{FN}})+p\,\delta_{\mathrm{FP}}
=
1-\delta_{\mathrm{FN}}-Jp
$$

이고, Bernoulli reward의 표준편차는

$$
\sigma(p)=\sqrt{m(p)\bigl(1-m(p)\bigr)}.
$$

정답 샘플과 오답 샘플이 받는 normalized reward의 조건부 평균은 각각

$$
\bar A_{\text{good}}
=
\frac{(1-\delta_{\mathrm{FN}})-m(p)}{\sigma(p)}
=
\frac{Jp}{\sigma(p)},
$$

$$
\bar A_{\text{bad}}
=
\frac{\delta_{\mathrm{FP}}-m(p)}{\sigma(p)}
=
-\frac{J(1-p)}{\sigma(p)}.
$$

따라서 good과 bad의 advantage gap은

$$
\bar A_{\text{good}}-\bar A_{\text{bad}}
=
\frac{J}{\sigma(p)}.
\tag{5}
$$

가 된다.

이 식의 의미는 논문의 핵심 중 하나다. 노이즈가 있다고 해서 gap이 복잡한 형태로 찢어지지 않는다. 정규화 이후에도 good과 bad의 상대적 분리는 정확히 \(J\)의 부호에 의해 결정된다. \(\sigma(p)\)는 오직 크기만 조절한다. 다시 말해, noisy reward는 방향(sign)과 속도(scale)를 분리해서 이해할 수 있다.

#### The Crucial Role of Youden’s Index.

식 (5)를 식 (4)에 대입하면, 학습의 부호는 \(J\) 하나로 요약된다. verifier가 random보다 낫다면 \((J>0)\) good 쪽으로 가는 drift가 생기고, verifier가 random과 같다면 \((J=0)\) drift가 사라지며, verifier가 random보다 나쁘다면 \((J<0)\) 오히려 bad 쪽으로 가는 drift가 생긴다.

여기서 중요한 점은 “노이즈가 크다”와 “노이즈가 해롭다”가 다르다는 것이다. 노이즈가 커도 \(J\)가 양수이면 학습은 여전히 가능하다. 반대로 총 노이즈율이 그리 크지 않더라도 \(J\)가 음수가 되면 시스템은 잘못된 방향으로 학습한다. 따라서 RLVR의 안전성은 절대 오차율이 아니라 순 판별력, 즉 \(J\)를 기준으로 판단해야 한다.

## 3 Phase Transition

binary setup에서 식 (5)를 식 (4)에 넣으면 bad mass에 대한 닫힌형태 ODE가 나온다.

$$
\dot p
=
-\eta \frac{J}{\sigma(p)}[p(1-p)]^2
\tag{6}
$$

이 식은 논문의 중심 방정식이다. reward의 세부 구현, good/bad 모드의 표면적 표현, 개별 completion의 문자열 차이를 모두 걷어낸 뒤에도, RLVR의 질적 거동은 결국 \(p\)와 \(J\)만으로 설명된다. 수많은 현상을 한 차원으로 눌러 담은 셈이다.

완전한 oracle에서는 \(\delta_{\mathrm{FN}}=\delta_{\mathrm{FP}}=0\), 따라서 \(J=1\)이고 \(m(p)=1-p\), \(\sigma(p)=\sqrt{p(1-p)}\)가 된다. 이때 식 (6)은

$$
\dot p=-\eta [p(1-p)]^{3/2}
\tag{7}
$$

로 단순화된다. clean case는 이후 noisy case를 해석하는 기준선 역할을 한다.

### 3.1 Bifurcation at the Critical Point

#### Fixed points and stability.

식 (6)의 경계 평형점은 \(p=0\)과 \(p=1\)이다. 각각 “모두 good”과 “모두 bad” 상태에 해당한다. 그런데 어느 쪽이 attractor가 되는지는 오로지 \(J\)의 부호에 의해 결정된다.

\(J>0\)이면 \(\dot p<0\)이므로 bad mass는 단조 감소하고 \(p=0\)이 안정점, \(p=1\)이 불안정점이 된다. 이 구간에서는 아무리 noisy해도 verifier가 평균적으로는 올바른 쪽을 더 많이 보상하므로, 학습은 결국 good 해로 간다.

\(J<0\)이면 부호가 완전히 뒤집힌다. 이제 \(\dot p>0\)가 되어 bad mass가 증가하고, \(p=1\)이 안정점이 된다. 즉, 학습은 더 많은 계산을 쓸수록 더 많이 틀리는 방향으로 진행된다. 논문이 “anti-learning”이라고 부르는 현상이 바로 이것이다.

\(J=0\)에서는 drift가 정확히 0이 되어 구간 전체가 중립 평형 집합이 된다. 이때는 학습이 일어나지 않는다. 이 세 경우가 만나는 지점이 바로 논문이 말하는 sharp phase transition이며, 경계는 \(J=0\)이다.

#### Special case: \(J=1\).

clean oracle에서는 식 (7)을 적분할 수 있고, bad mass 궤적의 닫힌식이 나온다. 논문의 식 (8)은 이 궤적을 step 수 혹은 연속시간 변수에 대해 직접 써 준다. 같은 내용을 implicit form으로 쓰면 다음과 같다.

$$
\frac{2p(t)-1}{\sqrt{p(t)(1-p(t))}}
=
\frac{2p(0)-1}{\sqrt{p(0)(1-p(0))}}
-\frac{\eta t}{2}.
\tag{8}
$$

이 식은 clean case에서 bad mass가 단순히 감소하는 것에 그치지 않고, 어떤 tail law로 감소하는지도 보여 준다. 특히 \(p\to 0\) 근방에서는 \(\dot p \sim -\eta p^{3/2}\)이므로

$$
p(t)=O(t^{-2})
$$

꼴의 다항식 감쇠가 나온다. 즉, 완전한 verifier 아래에서도 수렴은 무한히 빠른 지수형이 아니라, 경계 근방에서 자연스럽게 느려지는 power-law tail을 가진다.

#### RLVR limitation.

식 (8)이 드러내는 또 하나의 사실은 support barrier다. \(p=0\)이나 \(p=1\) 같은 경계는 흡수적이다. 더 일반적으로는 어떤 mode의 초기 확률이 정확히 0이면, multiplicative replicator flow 아래에서 그 mode는 이후에도 0을 유지한다. RLVR는 이미 존재하는 mode의 질량을 재분배하고 증폭할 수는 있지만, 초기 지지집합 밖의 mode를 새롭게 창조하는 데는 구조적 제약이 있다.

이 한계는 실무적으로 중요하다. base model이 어떤 프롬프트에 대해 정답 mode를 전혀 내지 못한다면, verifier가 아무리 좋아도 RLVR가 그 mode를 만들어 낼 신호 자체가 없다. 논문이 RLVR를 “기존 능력의 증폭기”로 보는 이유가 여기에 있다.

#### Asymptotics: which tail, and when.

논문은 “어느 쪽으로 가는가”와 “얼마나 빨리 가는가”를 분리한다. 방향은 \(J\)가 결정한다. 하지만 수렴 tail의 모양은 attractor 근처에서 reward variance가 남아 있는지, 아니면 사라지는지에 따라 달라진다.

\(J>0\)에서 attractor는 \(p=0\)이다. 이때 \(p=0\) 근처의 reward variance는 \(\delta_{\mathrm{FN}}\)에 의해 정해진다. 만약 \(\delta_{\mathrm{FN}}>0\)이면 정답만 남은 뒤에도 verifier가 정답을 일부 뒤집기 때문에 reward variance가 0으로 꺼지지 않는다. 이 경우 \(\sigma(0)>0\)이고 식 (6)은 근사적으로 \(\dot p\sim -c p^2\)가 되어

$$
p(t)=O(t^{-1})
$$

의 tail이 나온다. 반대로 \(\delta_{\mathrm{FN}}=0\)이면 \(p=0\)에서 reward variance도 같이 사라져 \(\sigma(p)\sim \sqrt p\)가 되고, 그 결과 clean case와 같은

$$
p(t)=O(t^{-2})
$$

tail이 나온다.

\(J<0\)에서는 attractor가 \(p=1\)로 바뀐다. 이 구간에서는 good mass \(1-p\)가 사라지는 방향으로 움직이며, 논문은 bad attractor 쪽에서는 learning regime에서 보았던 것과 같은 이중 분기보다 더 단순한 붕괴 구조가 나타난다고 정리한다. 핵심은 여전히 같다. 방향은 \(J\), tail law는 경계에서의 variance 구조가 정한다.

### 3.2 Rate, Not Fate

논문 제목의 핵심 구절이 바로 이 절에서 수학적으로 정리된다. \(J>0\)인 한, noisy dynamics와 clean dynamics는 같은 basin으로 수렴한다. 차이는 오직 시간축에 있다. 즉, noisy verifier가 만든 dynamics는 clean dynamics와 같은 궤적을 더 느리게 따라가는 것으로 볼 수 있다.

논문의 식 (9)은 이 사실을 state-dependent time reparameterization으로 표현한다. 요지는 간단하다. \(\sigma(p)\)가 크거나 \(J\)가 작으면 같은 \(p\) 변화량을 만들기 위해 더 많은 gradient step이 필요하다. 그 반대면 더 빨리 움직인다. 따라서 informative하지만 noisy한 verifier 아래에서는 계산량을 더 쓰는 것이 실제로 보상 역할을 할 수 있다.

하지만 이 명제는 \(J>0\)에서만 유효하다. \(J<0\)에서는 time-rescaling으로 clean case와 연결할 수 없다. 그때의 dynamics는 애초에 반대 방향으로 가기 때문이다. 그래서 논문은 “노이즈는 rate의 문제”라고 말하면서도, 동시에 “경계 \(J=0\)를 넘으면 fate의 문제”라고 분명히 선을 긋는다.

### 3.3 Maximal Learnability at Intermediate Bad Mass

식 (6)과 식 (7)은 어떤 프롬프트가 가장 빨리 배워지는지도 말해 준다. clean case에서 bad mass drift의 크기는 \([p(1-p)]^{3/2}\)에 비례하고, 이 값은 \(p=1/2\)에서 최대가 된다. 즉, 모델이 good과 bad를 반반 정도 섞어 내는 중간 난이도 프롬프트가 가장 학습 가능성이 크다.

반대로 \(p\)가 0에 가깝다면 이미 거의 맞히는 문제이므로 더 개선할 여지가 적다. \(p\)가 1에 가깝다면 거의 항상 틀리는 문제라서 정답 signal을 증폭할 여지가 적다. 중간 구간만이 signal과 variance가 동시에 충분한 영역이 된다.

#### Connection to prior “\(p(1-p)\)” learnability observations.

이 결과는 기존 연구의 경험적 관찰과도 맞닿아 있다. 중간 난이도 문제에서 progress가 가장 크다는 관찰은 여러 방식으로 보고되어 왔는데, 논문은 이를 GRPO mean-field dynamics에서 직접 도출한다. 단순히 reward variance가 중간에서 최대라는 설명을 넘어, bad mass를 줄이는 drift 자체가 중간 구간에서 최대가 된다는 점을 보여 준다.

노이즈가 비대칭이면 최적점은 \(p=1/2\)에서 벗어날 수 있다. 그 이유는 \(\sigma(p)\)가 bad mass에 따라 비대칭적으로 signal을 재가중하기 때문이다. 하지만 중요한 사실은 바뀌지 않는다. 가장 잘 배워지는 지점은 여전히 경계가 아니라 내부다. 학습은 “조금은 맞고 조금은 틀리는” 상태에서 가장 효율적이다.

## 4 LLM as a Multi-Armed Bandit

논문은 RLVR의 본질을 sequence-level decision problem으로 본다. 토큰 단위에서 보면 LLM 출력은 매우 복잡하지만, verifier가 보상을 주는 순간은 대개 전체 completion이 끝난 뒤다. 그러므로 이 setting에서는 토큰을 하나하나의 action으로 보는 것보다, 완성된 시퀀스 전체를 하나의 arm으로 보는 편이 자연스럽다.

### From sequences to modes.

물론 실제 시퀀스 공간은 매우 크다. 하지만 한 프롬프트 안에서는 완전히 다른 문자열이라도 같은 reasoning pattern을 공유하는 답들이 반복적으로 등장한다. 논문은 이런 반복 패턴을 reasoning mode로 묶는다. 그러면 completion 공간 \(\mathcal Y\)에서 mode 집합 \(\mathcal M\)으로 가는 coarse-graining map을 생각할 수 있다.

$$
g:\mathcal Y \to \mathcal M,\qquad
\pi_a=\Pr(g(y)=a)
$$

이제 정책은 거대한 문자열 분포가 아니라, 유한한 mode 위의 categorical distribution으로 바뀐다. 이 단순화는 단지 설명을 편하게 하려는 장치가 아니다. RLVR에서 중요한 것은 token-level stylistic variation이 아니라, verifier가 일관되게 보상하는 reasoning family가 무엇이냐이기 때문이다.

### Good vs. bad families.

mode들을 good family와 bad family로 나누면, 전체 확률벡터는 세 부분으로 분해된다. good family에 실린 총 질량은 \(1-p\), bad family에 실린 총 질량은 \(p\)다. 각 family 내부 조성은 정규화된 벡터 \(y\)와 \(z\)로 쓴다.

$$
\pi=((1-p)y,\; pz)
$$

여기서 \(y\)는 good mode들 안에서 질량이 어떻게 나뉘는지를, \(z\)는 bad mode들 안에서 질량이 어떻게 나뉘는지를 뜻한다. 이 분해는 논문 전체의 핵심 기법이다. 전체 성능은 주로 \(p\)가 결정한다. 하지만 어떤 정답 mode가 살아남는지, bad mode가 한 군데 몰리는지 여러 군데로 퍼지는지는 \(y\)와 \(z\)가 결정한다. 논문은 이 세 변수를 분리해 추적함으로써, 성능 변화와 mode selection을 동시에 설명한다.

이 관점에서는 프롬프트 하나가 사실상 작은 contextual bandit 문제로 바뀐다. arm은 reasoning mode이고, verifier는 각 arm에 noisy binary reward를 붙인다. GRPO는 평균보다 좋은 arm의 질량을 늘리고, 평균보다 나쁜 arm의 질량을 줄인다. 이때 bad mass가 줄어드는지 늘어나는지를 보는 것이 RLVR의 학습 성공 여부를 가장 직접적으로 보여 주는 상태변수가 된다.

## 5 Geometric Flow on the Probability Simplex

<figure>
  <img src="/assets/images/Rate_or_Fate_RLVeR/rlver_figure_04_geometry_of_the_probability_simplex.png" alt="Figure 4. Geometry of the probability simplex." />
  <figcaption><strong>Figure 4.</strong> Geometry of the probability simplex.</figcaption>
</figure>

Section 4에서 도입한 mode policy는 확률벡터이므로, 상태공간은 자연스럽게 simplex \(\Delta^{K+M-1}\)가 된다. 이 공간은 유클리드 벡터공간이 아니라, 합이 1로 보존되는 비유클리드 다양체다. 따라서 정책 업데이트를 제대로 이해하려면 단순한 좌표 증감이 아니라, 확률 단체 위의 기하를 봐야 한다.

softmax parameterization 아래에서 중요한 연산자는 Jacobian이다.

$$
J(\pi)=\mathrm{Diag}(\pi)-\pi\pi^\top
\tag{10}
$$

이 행렬은 세 역할을 동시에 한다. 첫째, softmax 미분 그 자체다. 둘째, 합이 0인 tangent space로 업데이트를 투영해 total mass 보존을 보장한다. 셋째, Shahshahani 혹은 Fisher 기하에서 natural gradient를 만들어 주는 inverse metric 역할을 한다.

이를 이용하면 GRPO mean-field flow는

$$
\dot\pi = \eta\,J(\pi)\,A
$$

로 쓸 수 있고, 각 좌표별로는 다음 replicator form을 얻는다.

$$
\dot\pi_i = \eta\,\pi_i\Bigl(A_i-\langle \pi, A\rangle\Bigr).
\tag{11}
$$

이 식은 아주 많은 것을 말해 준다. \(\pi_i=0\)이면 항상 \(\dot\pi_i=0\)이므로 simplex의 face는 불변이다. 또 각 arm은 절대 advantage가 아니라 population 평균에 비해 얼마나 나은지를 기준으로 증감한다. 이것이 RLVR가 상대 비교형 학습이라는 사실을 기하적으로 표현한 식이다.

### 5.1 Decoupling the Dynamics: Shape vs. Good and Bad Mass

\(\pi=((1-p)y,pz)\) 분해를 식 (11)에 넣으면, 전체 동역학이 shape dynamics와 mass dynamics로 분해된다. 정확한 식은 논문에서 (12a)–(12c)로 주어지며, 구조만 쓰면 다음과 같이 이해할 수 있다.

$$
\dot y_i
=
c(p)\,y_i\bigl(y_i-\|y\|_2^2\bigr),
$$

$$
\dot z_j
=
-c(p)\,z_j\bigl(z_j-\|z\|_2^2\bigr),
$$

$$
\dot p
=
-\eta \frac{J}{\sigma(p)}[p(1-p)]^2\bigl(\|y\|_2^2+\|z\|_2^2\bigr),
$$

여기서 \(c(p)\)는 \(J\), \(p\), \(\sigma(p)\)에 의해 정해지는 양의 스케일 함수이고, \(\|y\|_2^2\), \(\|z\|_2^2\)는 각 family 내부의 collision probability, 즉 concentration 정도를 뜻한다.

#### Interpretation.

이 분해는 세 가지 힘을 분리해서 보여 준다. 첫째, good block 내부에서는 \(y_i-\|y\|_2^2\)가 양수인 좌표가 더 커지고 음수인 좌표가 더 작아진다. 이는 self-reinforcing flow이며, good 질량이 한 arm에 집중되는 diversity collapse를 낳는다.

둘째, bad block 내부에서는 부호가 뒤집혀 concentration을 풀어 버리는 방향의 흐름이 생긴다. \(J>0\)에서 bad mass는 줄어들 뿐 아니라, 남아 있는 bad 질량도 한 점으로 몰리기보다 퍼지려는 경향을 가진다. 즉, 학습 regime에서는 good block은 sharpen되고 bad block은 diffuse된다.

셋째, 전체 bad mass \(p\)는 outer variable로서 good/bad 경쟁의 결과를 집계한다. 이 outer dynamics의 부호는 다시 \(J\)가 결정한다. 따라서 mode selection과 성능 개선이 분리되면서도, 동일한 기하적 틀 안에서 동시에 설명된다.

### 5.2 The right geometry: Shahshahani metric on \(\Delta\)

simplex 위에서는 확률이 작은 좌표를 조금 움직이는 것과, 큰 좌표를 같은 양만큼 움직이는 것이 같은 비용이 아니다. 희귀한 mode의 질량을 바꾸는 일은 상대적으로 큰 변화다. Shahshahani metric은 이 사실을 정확히 반영한다.

직관적으로는 각 tangent direction을 \(1/\pi_i\)로 재가중하는 기하라고 보면 된다. 이 기하 아래에서 Euclidean gradient를 그대로 쓰는 것이 아니라, softmax Jacobian을 통해 natural gradient로 변환한 방향이 가장 자연스러운 steepest-ascent 방향이 된다. 논문이 GRPO를 단순한 heuristic이 아니라 natural gradient flow로 해석하는 이유가 여기에 있다.

good block 내부에서는 이 흐름이 concentration potential 위의 ascent로 읽힌다. 다시 말해, GRPO는 good arm 내부에서 확률을 한 곳으로 집중시키는 potential을 따라 오르는 셈이다. 그래서 winner-take-all은 실험적 부작용이 아니라, simplex의 내재적 기하와 정규화 advantage가 만나서 생기는 구조적 결과가 된다.

#### KL-regularized mirror ascent and the replicator limit.

논문은 연속시간 ODE와 이산시간 mirror-ascent step이 같은 구조의 두 표현임도 보여 준다. KL-regularized mirror step은 다음 꼴로 생각할 수 있다.

$$
\pi^{+}
=
\arg\max_{q\in\Delta}
\Bigl\{
\langle A,q\rangle
-\frac{1}{\lambda}\mathrm{KL}(q\|\pi)
\Bigr\}
\tag{13}
$$

이 step을 작은 step size에서 전개하면, Section 5의 replicator flow와 일치한다. 즉, KL mirror ascent, Shahshahani natural gradient, replicator dynamics는 서로 다른 알고리즘이 아니라 하나의 기하적 구조를 다른 언어로 쓴 것에 가깝다.

### 5.3 Finite Sampling Cause Genetic Drift Noise

지금까지의 분석은 mean-field limit, 즉 rollout 수가 충분히 많아 기대값이 정확히 구현된다는 이상화 위에 서 있다. 실제 학습에서는 유한한 수의 샘플만 사용하므로, 각 그룹의 평균 reward와 표준편차는 noise를 가진 추정량이다.

논문은 이 유한 샘플 효과를 진화동역학의 genetic drift에 비유한다. 평균 drift 자체는 이전 절의 결정론적 ODE가 주지만, 그 위에 Wright–Fisher형 확산 노이즈가 덧씌워진다. score feature의 공분산이 softmax Jacobian과 같은 simplex 기하를 가지기 때문에, 이 샘플 노이즈도 같은 기하 속에서 해석된다.

중요한 것은 이 drift noise가 phase transition의 위치를 바꾸지 않는다는 점이다. 유한 샘플은 trajectory를 흔들 뿐이고, 1차 mean-field drift의 부호, 즉 \(J\)가 학습을 good 쪽으로 보내는지 bad 쪽으로 보내는지는 그대로 남는다. 실전에서 곡선이 들쭉날쭉해지는 이유와, 그럼에도 장기적 방향은 이론과 맞는 이유가 여기서 설명된다.

## 6 Mean-Field Dynamics of Bad Mass in GRPO

앞절까지의 논의는 REINFORCE형 mean-field skeleton을 중심으로 이루어졌다. 하지만 실제 GRPO는 importance sampling, PPO-style clipping, 그리고 상황에 따라 KL regularization까지 포함한다. 따라서 practical algorithm이 들어오면 앞서 본 phase transition이 바뀌는지 확인해야 한다. Section 6의 목적은 그 점을 분명히 하는 것이다.

### The role of clipping and importance sampling.

논문의 결론은 의외로 단순하다. 작은 step size와 fresh on-policy sampling을 가정하면, importance sampling과 ratio clipping은 leading-order drift를 바꾸지 않는다. 이들은 주로 \(O(\eta^2)\) 이하의 보정항으로 들어가며, 1차 mean-field phase portrait는 그대로 남는다.

이 결과를 반영해 논문은 multi-good/multi-bad setting에서 aggregate bad mass의 ODE를 다음과 같이 쓴다.

$$
\dot p
=
-\eta \frac{J}{\sigma(p)}[p(1-p)]^2
\Bigl(\|y\|_2^2+\|z\|_2^2\Bigr)
+O(\eta^2).
\tag{14}
$$

여기서 \(\|y\|_2^2\)와 \(\|z\|_2^2\)는 family 내부 concentration을 나타내는 collision mass다. good block이 한 arm에 더 집중될수록 \(\|y\|_2^2\)가 커지고, bad block이 퍼질수록 \(\|z\|_2^2\)는 작아진다. 따라서 multi-arm에서는 단순 binary case보다 drift에 “내부 형상”이 추가로 들어간다. 그러나 부호는 여전히 \(J\)가 잡는다.

#### Internal-time logit form.

논문은 \(p\) 대신 logit 변수

$$
\ell=\log\frac{p}{1-p}
$$

를 도입하고, 동시에 내부 시간(internal time) 변수를 바꾸면 drift 구조가 더 명확해진다고 설명한다. 이 좌표에서는 \(p(1-p)\) 같은 경계 포화 요인이 흡수되고, good–bad 경쟁의 방향이 더 직접적으로 드러난다. 핵심은 바뀌지 않는다. logit form으로 가도 deterministic drift의 부호는 여전히 \(J\)와 같고, \(J=0\)이면 leading-order drift가 사라진다.

#### Small-heterogeneity regime.

또한 good과 bad block 내부 조성이 거의 균등하다면, drift는 전체 벡터의 세부 구조 전체에 의존하기보다 \(\|y\|_2^2\)와 \(\|z\|_2^2\) 같은 저차 요약량에만 의존한다. 이것이 논문이 말하는 first-order geometry reduction이다. 즉, practical GRPO에서도 bad mass의 거동은 놀랍도록 저차원적이다.

#### Sign structure.

이 절의 결론은 분명하다. clipping과 importance sampling은 GRPO의 mean-field drift를 미세하게 다듬을 뿐, phase transition 자체를 옮기지 않는다. 학습이 가능한지, neutral한지, anti-learning인지의 구분은 여전히 \(J\)가 한다. 알고리즘 세부는 주로 속도, 안정성, finite-step 오차에 개입할 뿐이다.

### 6.1 KL Regularization: From Phase Transition to Interior Equilibrium

KL penalty를 넣으면 그림이 약간 달라진다. reward-driven drift에 reference policy로 되돌리려는 restoring drift가 추가되기 때문이다. 논문의 식 (18)은 이 regularized bad-mass ODE를 주고, 식 (19)는 내부 평형점에서 reward drift와 KL anchoring drift가 정확히 균형을 이루는 조건을 나타낸다.

핵심은 KL이 \(J\)의 부호를 바꾸는 것이 아니라, 경계 붕괴를 부드러운 interior equilibrium으로 바꾼다는 점이다. unregularized case에서는 \(J>0\)이면 \(p\to 0\), \(J<0\)이면 \(p\to 1\)로 간다. 그런데 KL이 있으면 어느 쪽이든 reference mass를 향한 당김이 생기므로, bad mass가 경계에 완전히 붙기 전에 내부에서 균형을 이룰 수 있다.

#### Unique interior equilibrium.

논문은 고정된 KL 세기와 reference mass가 주어졌을 때, 유일하고 전역적으로 안정한 interior equilibrium \(p^\star\in(0,1)\)가 존재함을 보인다. 그 위치는 \(J\)의 부호와 reference의 위치가 함께 결정한다. \(J>0\)이면 \(p^\star\)는 reference보다 더 good 쪽에 놓이고, \(J=0\)이면 정확히 reference mass에 놓이며, \(J<0\)이면 reference보다 bad 쪽으로 밀린다.

이 결과는 실무적으로도 중요하다. KL은 verifier가 anti-informative한데도 학습을 정답 쪽으로 바꿔 주는 장치가 아니다. 그보다는 잘못된 verifier가 만들어 내는 붕괴를 완화하고, 확률 질량이 극단적인 경계에 달라붙지 않도록 막는 안정화 장치다.

#### Asymptotic regimes.

KL이 매우 강하면 equilibrium은 reference policy에 가깝게 붙는다. 반대로 KL이 매우 약하면 equilibrium은 원래 reward-driven boundary에 가까워진다. 하지만 KL이 0이 아닌 한, collapse는 완전히 경계까지 가지 않고 내부에서 멈춘다. 논문은 이 현상을 “phase transition의 smoothing”으로 해석한다.

<figure>
  <img src="/assets/images/Rate_or_Fate_RLVeR/rlver_figure_05_kl_regularization_smooths_the_phase_transition.png" alt="Figure 5. KL regularization smooths the phase transition." />
  <figcaption><strong>Figure 5.</strong> KL regularization smooths the phase transition.</figcaption>
</figure>

Figure 5는 바로 이 smoothing을 시각적으로 보여 준다. unregularized system에서 \(J=0\)은 학습과 붕괴가 갈리는 날카로운 경계였지만, KL이 들어가면 그 경계는 interior fixed point를 갖는 연속적인 구조로 변한다. 그렇다고 해서 \(J\)의 역할이 사라지는 것은 아니다. KL은 방향을 바꾸지 않고, 단지 끝점을 안쪽으로 끌어온다.

## 7 Experiments

논문의 실험 목표는 매우 분명하다. 첫째, 실제 GRPO 학습에서도 \(J=0\)에서 sharp phase transition이 나타나는가. 둘째, \(J>0\) 구간에서 noise는 truly rate만 바꾸고 fate는 바꾸지 않는가. 이론은 단순한 mean-field ODE에서 나온 것이므로, 실제 LLM 코드 학습에서 이 구조가 유지되는지 검증하는 것이 중요하다.

### 7.1 Experimental Hypotheses

논문은 두 가지 가설을 명시한다. 첫째, **Phase Transition at \(J=0\)**: \(J>0\)일 때만 accuracy가 체계적으로 개선되고, \(J=0\)에서는 중립 표류가 되며, \(J<0\)에서는 성능이 악화되어야 한다. 둘째, **Rate, not Fate**: \(J>0\)인 한, 노이즈가 커질수록 수렴은 느려지지만 basin of attraction은 바뀌지 않아야 한다.

즉, 실험은 단순히 “노이즈가 얼마나 성능을 깎는가”를 보는 것이 아니라, 곡선의 방향 자체가 어느 지점에서 뒤집히는지, 그리고 learning regime 안에서 서로 다른 noise setting이 같은 방향의 trajectory를 유지하는지를 보려는 것이다.

### 7.2 Setup

#### Task and data.

실험 과제는 Python 코드 생성이다. 각 데이터 포인트는 자연어 문제 설명, 입출력 예시, public/hidden test harness를 포함한다. 이 선택은 단순히 LLM이 잘하는 과제이기 때문이 아니라, verifier noise가 실제로 자연스럽게 생기는 과제이기 때문이다.

#### Model and evaluation.

기본 정책은 Qwen2.5-3B이며, 지표는 validation pass@1이다. 각 설정은 다섯 개의 독립 실행으로 반복되어 평균과 표준편차를 계산한다. 이 설정은 noisy trajectory의 평균적 방향과 run-to-run 분산을 함께 보기 위한 것이다.

#### Training algorithm.

학습은 VeRL 기반의 표준 GRPO를 사용한다. 프롬프트당 8개의 rollout을 샘플링하고, 그룹 내 reward를 평균 0, 표준편차 1로 정규화한 다음 PPO-style importance ratio clipping을 적용한다. 이론에서 본 reward-driven 구조만 보기 위해 KL penalty coefficient는 0으로 둔다.

#### Synthetic verifier noise.

oracle checker가 주는 true correctness를 먼저 얻은 뒤, 그 결과를 Bernoulli flip으로 뒤집어 operational reward를 만든다. 정답이면 \(\delta_{\mathrm{FN}}\) 확률로 실패 처리하고, 오답이면 \(\delta_{\mathrm{FP}}\) 확률로 성공 처리한다. 이 구현은 Appendix M의 pseudocode에 명시되어 있다. 같은 목표 \(J\)에 대해 서로 다른 FP/FN 조합을 시험해, 단순한 signal strength와 error type asymmetry를 분리해서 본다.

#### Protocol.

각 noise condition마다 모델을 2 epoch, 총 1,410 step 동안 학습하고 5 step마다 지표를 기록한다. 모든 noise setting에서 나머지 하이퍼파라미터는 고정한다. 즉, 관측된 차이는 verifier noise에서만 나오도록 설계되어 있다.

#### Baseline.

기준선은 clean oracle, 즉 \((\delta_{\mathrm{FP}},\delta_{\mathrm{FN}})=(0,0)\)인 GRPO다. 이것이 perfect verification 아래에서의 상한선 역할을 한다.

### 7.3 Results

<figure>
  <img src="/assets/images/Rate_or_Fate_RLVeR/rlver_table_01_validation_accuracy_noise_sensitivity.png" alt="Table 1. Validation accuracy and noise sensitivity." />
  <figcaption><strong>Table 1.</strong> Validation accuracy and noise sensitivity.</figcaption>
</figure>

#### Phase transition confirmed (\(\mathcal{H}_1\)).

Table 1과 Figure 1은 \(J=0\)에서 질적 경계가 생긴다는 가설을 지지한다. noise-free oracle \((0.00,0.00)\)에서는 2 epoch 후 validation pass@1이 20.8%였고, base model 대비 +8.0% 개선되었다. 비교적 강한 learning regime인 \((0.20,0.10)\), 즉 \(J=0.70\)에서는 pass@1이 18.6%로 +5.8% 개선되었다. verifier가 다소 흐려져도 \(J\)가 충분히 양수이면 학습은 계속 일어난다는 뜻이다.

\(J=0.30\)인 두 설정도 이를 뒷받침한다. \((\delta_{\mathrm{FP}},\delta_{\mathrm{FN}})=(0.00,0.70)\)에서는 15.98%, \((0.70,0.00)\)에서는 14.64%가 보고된다. 둘 다 여전히 learning regime이지만, clean case보다는 확실히 느리고 낮다. 즉, basin은 유지되되 속도가 줄어든다.

반면 \(J=0\)인 \((0.50,0.50)\)에서는 개선이 +0.6%에 그친다. 이는 이론의 neutral drift와 맞아떨어진다. verifier가 good과 bad를 구분하는 순정보를 전혀 주지 못하므로, 학습이 체계적 방향을 얻지 못하는 것이다.

가장 인상적인 결과는 \(J<0\)인 \((0.60,0.50)\) 설정이다. 이 경우 base model 대비 -12.6%의 성능 악화가 나타난다. 단순히 학습이 멈추는 것이 아니라, 반복 업데이트가 실제로 bad equilibrium 쪽으로 시스템을 밀어 버린다. 논문이 말하는 “anti-learning”이 실험에서도 그대로 관측된 셈이다.

#### Noise rescales speed, not fate (\(\mathcal{H}_2\)).

\(J>0\) 조건들만 놓고 보면, validation accuracy 곡선은 모두 같은 방향으로 움직인다. clean oracle이 가장 빠르고, \(J\)가 작아질수록 개선 속도와 최종 도달 성능이 낮아진다. 하지만 학습 방향 자체는 여전히 위쪽이다. 이 점은 Section 3의 time-rescaling 해석과 잘 맞는다.

또한 같은 \(J\)라도 FP와 FN의 배치가 성능을 바꾼다는 점이 중요하다. \(J=0.30\)으로 같아도 \((0.00,0.70)\)가 \((0.70,0.00)\)보다 더 높다. 이는 coding RLVR 맥락에서 false positive가 false negative보다 더 해로울 수 있음을 시사한다. 정답을 놓치는 것은 이미 존재하는 good mode의 증폭 속도를 늦추는 문제지만, 오답을 정답으로 보상하는 것은 bad mode 자체를 적극적으로 강화하기 때문이다.

이 결과는 논문의 이론과도 잘 연결된다. \(J\)는 부호를 정하고, FP/FN의 구체적 배치는 \(\sigma(p)\)와 경계 분산 구조를 통해 속도와 비대칭성을 조정한다. 즉, fate는 \(J\), finer rate effects는 noise geometry가 맡는다는 구도가 실험에서도 드러난다.

### 7.4 Limitations and Future Directions

#### Oracle imperfection.

실험에서 “oracle”이라고 부르는 checker도 실제로는 유한한 테스트 세트에 의존한다. 따라서 edge case에서는 true correctness 추정이 완전히 정확하지 않을 수 있다. 이 한계는 coding RLVR에서 피하기 어렵고, 오히려 논문 주제와 직접 맞닿아 있다.

#### Context length effects.

성능이 낮은 설정에서는 장황하고 불안정한 답이 늘어나면서 truncation이 증가할 수 있다. VeRL이 truncation을 reward 0으로 다루면, 이것은 사실상 추가 false negative를 넣는 효과가 된다. 논문은 anti-learning 구간에서의 관측 비대칭 일부가 이런 시스템적 요인에서 올 수 있다고 본다.

#### Generalization.

실험은 Python 코드와 Qwen2.5-3B에 집중되어 있다. 저자들은 \(J=0\) 경계 자체는 보다 일반적인 group-normalized RLVR의 구조일 것으로 보지만, 정확한 decay rate와 noise tolerance는 모델 크기, 과제 복잡도, verifier 형태에 따라 달라질 수 있다고 본다.

#### Time-dependent noise.

본문 분석은 \(\delta_{\mathrm{FP}},\delta_{\mathrm{FN}}\)이 시간에 따라 변하더라도 “그 순간의 instantaneous drift”는 여전히 식 (6)로 해석할 수 있다는 점을 암시한다. 그러나 실험은 고정된 노이즈율만 다룬다. policy와 verifier가 함께 진화하는 co-evolution setting은 이론적으로도, 실험적으로도 다음 단계 과제로 남는다.

## 8 Conclusion

논문의 결론은 아주 압축적으로 말할 수 있다. RLVR에서 noisy verifier가 문제인지 아닌지는 “노이즈가 있는가”가 아니라 “그 verifier가 여전히 random보다 나은가”로 판단해야 한다. 그 판단을 담당하는 스칼라가 \(J=\mathrm{TPR}-\mathrm{FPR}\)다. \(J>0\)이면 학습은 성립하고, \(J=0\)이면 중립화되며, \(J<0\)이면 학습은 역방향으로 뒤집힌다.

이 결과는 RLVR를 이해하는 관점을 바꾼다. 기존에는 noisy label이 있어도 데이터가 충분하면 평균적으로 극복할 수 있다고 생각하기 쉽다. 그러나 이 논문은 verifier가 anti-informative한 구간으로 들어가면, 평균을 더 많이 낼수록 오히려 잘못된 방향을 더 세게 학습한다고 보여 준다. 이때는 더 많은 rollout과 더 긴 학습이 해법이 아니라 악화 요인이다.

### What this paper contributed.

첫째, completion들을 reasoning mode로 묶는 multi-armed bandit 추상화를 제시했다. 둘째, GRPO를 probability simplex 위의 replicator / natural-gradient flow로 해석했다. 셋째, 전체 dynamics를 outer bad-mass evolution과 inner within-block competition으로 분해했다. 넷째, \(J=0\)에서 sharp phase transition이 생긴다는 사실을 mean-field ODE로 보였다. 다섯째, learning regime에서는 noisy reward가 rate를 바꾸되 fate를 바꾸지 않는다는 “rate, not fate” 결과를 제시했다. 여섯째, clipping, importance sampling, KL regularization 같은 practical 요소가 이 구조에 어떻게 개입하는지 정리했다.

### Closing perspective.

RLVR는 verifier가 조금만 좋아도 무조건 잘 되는 알고리즘이 아니다. verifier가 random보다 낫기만 하면 long-run basin은 유지되지만, 그 경계는 매우 명확하고 차갑다. \(J\)가 0을 넘느냐 넘지 못하느냐가 학습과 붕괴를 가른다. 따라서 RLVR 설계에서 가장 먼저 물어야 할 질문은 “rollout을 얼마나 더 늘릴까”가 아니라 “현재 verifier의 \(J\)가 정말 양수인가”이다.

## References

[원문 References 삽입]

## Appendix A LLM as Multi-arm Bandit

### A.1 Multi-armed bandits.

부록 A는 본문에서 사용한 bandit 비유를 엄밀하게 정리한다. multi-armed bandit 문제에서는 각 라운드마다 하나의 arm을 선택하고, 그 arm에 대한 확률적 reward를 관측한다. 목표는 미지의 reward 분포 아래에서 누적 보상을 최대화하는 것이다. RLVR와 완전히 같은 문제는 아니지만, “행동 하나를 고르고 그 결과에 대한 noisy scalar feedback를 받는다”는 통계 구조는 매우 유사하다.

논문이 bandit을 가져오는 이유는 토큰 수준의 복잡도를 걷어 내기 위해서다. RLVR에서 verifier는 보통 completion이 끝난 뒤에만 신호를 주므로, token-level MDP보다 sequence-level bandit이 이론화에 더 자연스럽다.

### A.2 Bandit Abstraction for LLMs

길이 제한 \(T\)를 두면, 한 프롬프트에서 가능한 completion의 effective support는 유한 집합으로 볼 수 있다. 그러면 각 completion을 하나의 arm처럼 취급할 수 있고, 정책은 이 arm들 위의 categorical distribution이 된다. 이 추상화는 실제 language modeling의 모든 세부를 살리지는 않지만, verifier가 시퀀스 수준 reward를 준다는 RLVR의 핵심 구조는 보존한다.

### A.3 Coarse-graining into Modes

개별 문자열을 arm으로 두면 support가 여전히 너무 크다. 그래서 논문은 의미상 같은 reasoning path를 하나의 mode로 묶는 coarse-graining을 도입한다. 수학 풀이에서는 같은 논리 전개를 공유하는 서로 다른 chain-of-thought가, 코드 문제에서는 같은 알고리즘을 구현한 서로 다른 코드가 하나의 mode에 대응한다.

이 coarse-graining 이후 정책은 mode 위의 확률분포가 되며, bad mass \(p\)가 자연스럽게 등장한다. 본문과 이후 부록의 대부분은 이 mode-level representation 위에서 전개된다.

## Appendix B Noisy Rewards and Youden’s \(J\) Index

<figure>
  <img src="/assets/images/Rate_or_Fate_RLVeR/rlver_figure_06_reward_variance_geometry_under_noisy_bernoulli_rewards.png" alt="Figure 6. Reward variance geometry under noisy Bernoulli rewards." />
  <figcaption><strong>Figure 6.</strong> Reward variance geometry under noisy Bernoulli rewards.</figcaption>
</figure>

부록 B는 noisy Bernoulli reward의 평균과 분산을 명시적으로 계산해, 본문의 “\(J\)가 모든 것을 정한다”는 주장을 확률 계산으로 뒷받침한다. bad mass가 \(p\)일 때 noisy reward의 평균은

$$
m(p)=
(1-p)(1-\delta_{\mathrm{FN}})+p\delta_{\mathrm{FP}}
=
1-\delta_{\mathrm{FN}}-Jp
\tag{20}
$$

이다. reward가 Bernoulli이므로 분산은

$$
\mathrm{Var}(r)=m(p)\bigl(1-m(p)\bigr)
\tag{21}
$$

이며, 식 (22)는 이를 직접 전개해 같은 결과가 나온다는 점을 확인한다.

그 다음 논문은 z-score normalization을 적용한 뒤 정답 arm과 오답 arm의 조건부 평균 normalized reward를 계산한다. 결과는

$$
\mathbb E[\tilde r\mid \text{good}]
=
\frac{Jp}{\sigma(p)},
\tag{23}
$$

$$
\mathbb E[\tilde r\mid \text{bad}]
=
-\frac{J(1-p)}{\sigma(p)},
\tag{24}
$$

$$
\mathbb E[\tilde r\mid \text{good}]
-
\mathbb E[\tilde r\mid \text{bad}]
=
\frac{J}{\sigma(p)}
\tag{25}
$$

가 된다. 본문 식 (5)가 바로 이 결과다. 정규화가 해도 되는 일과 하지 못하는 일이 여기서 분명해진다. 정규화는 reward scale을 바꾸고, \(\sigma(p)\)를 통해 속도에 개입할 수는 있다. 하지만 signal의 방향은 새로 만들지 못한다. 방향은 오직 \(J\)가 정한다.

부록 B의 여러 remark는 centered-only normalization이나 \(\{-1,+1\}\) reward parameterization처럼 표면적 구현이 달라져도 핵심 부호 구조가 유지된다는 점을 덧붙인다. 결국 normalized reward 아래에서도 “good이 average보다 위에 있는가, bad가 average보다 위에 있는가”를 정하는 것은 \(J\)뿐이다.

## Appendix C Mean Field Dynamics

부록 C는 multi-good/multi-bad setting에서 mean-field ODE를 자세히 유도한다. 핵심은 logit 공간에서 기대 업데이트를 먼저 계산하고, 그 다음 softmax Jacobian을 통해 simplex 위의 확률 dynamics로 밀어 넣는 것이다. 여기서 rank-one coupling term을 빠뜨리면 zero-sum constraint가 깨져 전체 질량 보존이 사라지므로, 이 부록은 기하학적 일관성을 유지하는 데 중요하다.

### C.1 Dynamics of the Bad Arms

bad arm 각각의 drift는 자기 자신만이 아니라 good block 전체와의 상대 advantage에 의해 정해진다. 중요한 것은 bad arm들을 모두 합한 total bad mass drift다. 부록의 식 (34)는 이 total drift가 block 내부 collision structure에 의해 조정된다는 사실을 보여 준다. 즉, bad arm이 많다고 해서 무조건 bad mass가 빨리 줄거나 늘어나는 것이 아니라, 그 bad mass가 내부적으로 어떻게 퍼져 있느냐가 함께 영향을 준다.

#### Total Bad-Mass Drift

식 (34)의 요지는 본문 식 (14)와 같다. total bad mass의 drift는 \(J/\sigma(p)\)에 비례하고, 추가로 \(\|y\|_2^2+\|z\|_2^2\) 같은 geometry factor가 붙는다. 따라서 multi-arm에서는 good 내부 specialization과 bad 내부 diffusion이 outer bad-mass dynamics의 속도에 직접 연결된다.

#### Within-Bad Dynamics in Normalized Coordinates

bad block 내부 정규화 좌표 \(z\)로 가면, dynamics는 collision field와 반대 부호의 feedback로 쓸 수 있다. \(J>0\)에서는 bad block이 uniform 분포 쪽으로 퍼지고, \(J<0\)에서는 부호가 뒤집혀 bad block 내부에서도 winner-take-all collapse가 생긴다. 이는 Appendix J에서 더 자세히 분석된다.

### C.2 Dynamics of the Good Arms

good block 내부 정규화 좌표 \(y\)에서는 부호가 반대로 나타난다. \(J>0\)일 때 good block은 uniform mixture를 떠나 한 arm에 집중되는 방향으로 움직인다. 이는 단순한 경험적 관찰이 아니라, 부록에서 Jacobian과 collision term을 통해 직접 유도된다.

결국 good block에서는 specialization, bad block에서는 de-concentration이 생기고, 이 둘이 outer bad-mass 감소와 결합해 본문 Figure 2의 구조를 만든다.

### C.3 From Expectation-Based Updates to ODEs: The Small-Step Bridge

이 절은 이산시간 gradient update와 연속시간 ODE 사이의 연결을 정리한다. 학습률 \(\eta\)가 충분히 작으면, 한 step의 기대 업데이트는 연속시간 drift의 Euler step처럼 해석할 수 있다.

#### Option 1: Unit Time per Iteration

한 번의 update를 시간 1로 두면, step index가 곧 시간 변수가 된다. 이 해석은 알고리즘 step 수와 ODE 시간을 바로 연결할 때 편리하다.

#### Option 2: Alternative Time Rescaling

반대로 \(\eta\)나 \(p\)-의존 prefactor를 시간축으로 흡수하는 방식도 가능하다. 논문은 이 두 시간척도가 서로 다른 trajectory를 만드는 것이 아니라, 같은 궤적을 다른 속도로 보는 것임을 강조한다. Section 3.2의 “rate, not fate” 해석도 사실 이 시간 재매개화 관점과 연결된다.

## Appendix D Maximal Learnability

<figure>
  <img src="/assets/images/Rate_or_Fate_RLVeR/rlver_figure_07_learnability_maximizing_bad_mass.png" alt="Figure 7. Learnability-maximizing bad mass." />
  <figcaption><strong>Figure 7.</strong> Learnability-maximizing bad mass.</figcaption>
</figure>

부록 D는 “어떤 프롬프트가 가장 빨리 학습되는가”를 정량화한다. 논문은 이를 bad mass의 instantaneous drift 크기로 정의하고, learnability speed \(L(p)\)를 도입한다. multi-arm 설정에서도 핵심 구조는

$$
L(p)\propto \frac{[p(1-p)]^2}{\sigma(p)}
\tag{45–47의 요지}
$$

로 요약된다. 이 값이 클수록 한 step당 bad mass가 더 많이 줄어든다.

### Normalized separation under noisy rewards.

z-score normalized reward를 쓰면 good과 bad의 separation은 \(J/\sigma(p)\)가 된다. 따라서 learnability는 단순히 \(J\)만으로 정해지지 않고, \(\sigma(p)\)를 통해 현재 난이도 상태 \(p\)와도 상호작용한다.

### Noiseless case (\(J=1\)).

clean oracle에서는 \(\sigma(p)=\sqrt{p(1-p)}\)라서

$$
L(p)\propto [p(1-p)]^{3/2}
$$

가 되고, 최대점은 \(p=1/2\)다. 이는 본문 Section 3.3의 결과를 더 일반적인 형태로 다시 보여 준다. 너무 쉬운 문제도, 너무 어려운 문제도 느리게 학습되고, 중간 난이도 문제가 가장 잘 배워진다.

### Noisy grading: what changes.

대칭 노이즈라면 최적점은 여전히 중간 근처에 남는다. 하지만 비대칭 노이즈에서는 \(\sigma(p)\)가 비대칭적으로 signal을 재가중하므로 최적 bad mass가 \(1/2\)에서 이동한다. 예를 들어 false positive가 두드러지면, bad mass가 조금만 많아져도 verifier가 오답을 과하게 통과시키므로 최적점이 다른 내부 지점으로 옮겨 간다.

이 절은 Figure 7을 통해 “중간 난이도가 가장 학습 가능하다”는 명제가 단지 감각적인 말이 아니라, noisy RLVR에서도 계산 가능한 구조라는 점을 보여 준다.

## Appendix E Lyapunov analysis and the role of \(J\)

부록 E는 본문의 phase transition을 Lyapunov 함수와 전역 안정성으로 다시 증명한다. 단순히 two-state ODE만 보는 것이 아니라, multi-arm simplex 전체에서 attractor가 어떻게 바뀌는지를 보여 주는 부분이다.

### Block symmetry and GRPO parametrization.

good arm과 bad arm 각각에 대해 block symmetry를 가정하면, advantage gap과 mass transfer를 더 간단한 형태로 정리할 수 있다. 이때 \(J\)는 good block과 bad block 사이의 전체 정렬 방향을 나타내는 파라미터가 된다.

### GRPO mean-field flow.

부록의 Theorem E.1은 적절한 potential과 Lyapunov 함수를 구성해, \(J>0\)이면 good face가 전역 attractor, \(J<0\)이면 bad face가 전역 attractor, \(J=0\)이면 simplex 전체가 equilibrium 집합이 된다는 사실을 보인다. 이는 본문 Section 3의 직관을 전역적, 다차원적으로 일반화한 결과다.

또한 이 부록은 tail behavior에 대한 정량적 bound도 함께 준다. 즉, phase transition은 단순한 그림 설명이 아니라, 실제로 potential 함수의 단조성으로 증명되는 구조라는 점을 강조한다.

### Decomposition dynamics and Shahshahani structure.

\((y,z,p)\) 분해에서 보면, good block 내부 potential은 ascent 흐름을 따르고 bad block 내부 potential은 부호가 뒤집힌 흐름을 따른다. 따라서 \(J>0\)에서는 good block이 점점 더 concentrate되고 bad block은 더 uniform해진다.

### Probabilistic interpretation of the potential \(\Phi(y)\)

good block potential \(\Phi(y)\)는 사실상 \(\|y\|_2^2\), 즉 collision probability 혹은 Herfindahl concentration과 연결된다. 값이 클수록 질량이 소수 arm에 집중되어 있다는 뜻이다. 따라서 \(J>0\)에서 \(\Phi(y)\)가 증가한다는 것은, 학습이 진행될수록 good mode 중 하나가 승자가 된다는 의미다.

### Coordination-game correspondence.

논문은 이 dynamics를 대칭 coordination game의 replicator dynamics와도 연결한다. 각 arm의 payoff를 현재 점유율에 연결하면, RLVR의 within-good competition은 coordination game에서 한 전략이 선택되는 과정과 같은 수학 구조를 가진다. 이 비교는 winner-take-all 현상이 우연한 구현 artifact가 아니라는 점을 다시 보여 준다.

## Appendix F Bad arm dynamics under PPO/GRPO style importance sampling and clipping

부록 F는 PPO/GRPO 구현에서 거의 항상 등장하는 importance sampling과 clipping이 왜 leading-order mean-field drift를 바꾸지 않는지 직접 계산한다.

### Setup.

old policy에서 샘플링하고 new policy에서 gradient를 취하면 importance ratio가 생긴다. PPO는 이 비율이 너무 커지지 않게 clipping을 건다. 직관적으로는 이 요소들이 Section 3의 nice ODE를 망가뜨릴 것 같지만, small-step regime에서는 그렇지 않다.

### IS score-function update.

importance ratio를 곱한 score-function update를 전개하면 추가항이 생긴다. 하지만 이 항은 ratio가 1 주변에서 움직이는 작은 양이므로, \(\eta\)의 1차항이 아니라 2차항 이하로 밀려난다. 그래서 평균 drift의 부호 구조는 본문과 동일하게 유지된다.

### Expanded form and relation to Appendix C.

softmax pushforward까지 포함해 정리하면, IS가 만든 extra term은 simplex 위에서 결국 더 높은 차수의 잔차항으로 들어간다. Appendix C의 REINFORCE dynamics가 practical GRPO의 leading-order skeleton이라는 말이 여기서 증명된다.

### Block-symmetric specialization.

block symmetry를 넣으면 이 사실이 더욱 명확해진다. good/bad aggregate dynamics는 그대로 남고, practical 구현이 만드는 차이는 세부 수치와 finite-step stabilization에 제한된다.

### Clipping in PPO and GRPO.

clipping은 ratio가 threshold를 넘을 때만 활성화된다. 작은 step size에서는 ratio가 대부분 1 근처에 있으므로, clipping 자체가 거의 작동하지 않는다. 그래서 mean-field limit에서는 clipping도 부호 구조를 바꾸지 못한다.

### Conclusion.

결국 PPO/GRPO의 실전 요소는 Section 6의 ODE를 무효화하지 않는다. phase transition은 알고리즘 디테일을 넘어서는 구조적 결과다.

## Appendix G KL Regularization

부록 G는 KL regularization을 보다 엄밀하게 다룬다. Section 6.1이 결과를 요약했다면, Appendix G는 어떤 KL을 쓰느냐에 따라 drift가 어떻게 달라지는지 상세히 계산한다.

### Reference policy.

현재 정책과 마찬가지로 reference policy도 good/bad mass와 block 내부 조성으로 분해할 수 있다. 이렇게 해야 KL이 total bad mass에만 작용하는지, within-block 조성까지 끌어당기는지 구분할 수 있다.

### Replicator/natural-gradient form (recall).

KL penalty를 natural-gradient language로 쓰면, 이는 simplex 위에서 특정 reference 쪽으로 가는 additional potential term으로 해석된다. reward-driven ascent 위에 reference-anchoring drift가 겹쳐지는 구조다.

### Two KL choices.

논문은 두 가지 KL을 구분한다. 하나는 good vs. bad 두 클래스만 보는 inter-block KL이고, 다른 하나는 block 내부 조성까지 포함하는 full reverse-KL이다. 전자는 total bad mass \(p\)에만 직접 작용하고, 후자는 \(p\)뿐 아니라 \(y,z\)의 내부 조성도 reference 쪽으로 당긴다.

### G.1 Full Mean-Field ODE for the Bad Mass with KL

reward term에 KL drift를 더하면 Section 6.1의 regularized ODE가 나온다. 여기서 본질은 KL이 reward term을 없애는 것이 아니라, additional restoring force를 더한다는 점이다.

### G.2 Nullcline, Interior Equilibrium, and the Prevention of Collapse

부록 G는 interior nullcline을 정의하고, 그 위에 유일한 interior equilibrium \(p^\star\)가 존재함을 보인다. \(J<0\)라도 KL이 0이 아니면 system이 bad vertex에 완전히 달라붙지 않고 내부에서 멈출 수 있다. 이것이 “collapse prevention”이다.

#### Asymptotic behavior of \(p^\star\)

강한 KL에서는 \(p^\star\)가 reference mass에 붙고, 약한 KL에서는 원래 reward-driven boundary에 가까워진다. 따라서 KL의 역할은 verifier의 신호를 정정하는 것이 아니라, 극단적 붕괴를 완충하는 것이라고 이해하는 것이 정확하다.

## Appendix H Properties of the Simplex

부록 H는 논문 전체에서 반복해서 사용한 simplex 기하를 정리한다.

### Geometric and information-theoretic intuition.

simplex는 단순한 삼각형이나 다면체가 아니라, 범주분포 family의 정보기하가 사는 공간이다. square-root embedding으로 보면 이는 단위구의 양의 orthant와 연결되고, KL과 Fisher metric이 이 공간의 자연스러운 거리 개념이 된다.

### Riemannian gradient under the Shahshahani metric.

Shahshahani metric 아래의 natural gradient는 Euclidean gradient에 softmax Jacobian을 곱한 형태가 된다. 따라서 Section 5의 \(J(\pi)=\mathrm{Diag}(\pi)-\pi\pi^\top\)는 단순한 미분 행렬이 아니라, 확률공간의 자연기하를 구현하는 핵심 객체다.

### Remark on boundary points.

일부 좌표가 0인 경계에서도 같은 논의는 남아 있는 support face 위에서 그대로 유효하다. 이는 왜 zero-probability mode가 다시 살아나지 않는지를 기하적으로 설명해 준다.

### Closing the loop.

결국 mirror ascent, natural gradient, replicator flow는 하나의 구조를 서로 다른 표현으로 나타낸 것이다. 부록 H는 논문 전체의 기하학적 해석을 정리하는 역할을 한다.

## Appendix I Inner Dynamics of the Good Arms

부록 I는 good block 내부 조성 \(y\)의 dynamics를 자세히 분석한다. 본문이 total bad mass \(p\)에 집중했다면, 이 부록은 “good mode들 사이에서 결국 누가 살아남는가”를 다룬다.

### I.1 Evolution of Collision term, \(s_2\)

good block의 collision term을

$$
s_2=\sum_i y_i^2
$$

로 두면, \(J>0\)일 때 \(s_2\)는 증가한다. 이는 good block 내부 질량이 점점 덜 균등해지고, 한 arm에 더 집중된다는 뜻이다. \(s_2\)가 커질수록 concentration이 커지고, uniform mixture에서 멀어진다.

### I.2 Asymptotic General inner dynamics behavior

논문은 generic initial condition, 즉 동률이 없는 초기화에서는 초기에 가장 큰 good arm이 끝까지 우세를 유지하고, 결국 vertex로 수렴함을 보인다. 이 결과는 Figure 2의 winner-take-all 구조를 수학적으로 뒷받침한다.

### I.3 Stability of the Within–Good Equilibria

good block 내부에서는 uniform equilibrium과 pure-arm vertex equilibrium이 주요 평형점이다. \(J\)의 부호에 따라 이들의 안정성이 서로 뒤바뀐다.

<figure>
  <img src="/assets/images/Rate_or_Fate_RLVeR/rlver_table_02_stability_of_uniform_and_pure_arm_equilibria.png" alt="Table 2. Stability of uniform and pure-arm equilibria." />
  <figcaption><strong>Table 2.</strong> Stability of uniform and pure-arm equilibria.</figcaption>
</figure>

Table 2의 내용은 간단하지만 중요하다. \(J>0\)이면 uniform equilibrium은 불안정하고, pure-arm vertex는 안정하다. 따라서 diversity collapse와 specialization이 일어난다. 반대로 \(J<0\)이면 uniform equilibrium이 안정하고 vertex가 불안정해져, good block 내부 다양성이 유지된다. 즉, verifier가 informative할 때에만 GRPO는 “한 정답 모드로의 집중”을 유도한다.

### I.4 Coupling back to physical time

internal time에서는 \(y\) dynamics가 자율적이지만, 실제 물리적 시간에서는 \(p(t)\)와 결합된다. bad mass가 얼마나 빨리 줄어드느냐에 따라 good block 내부 polarization이 실제 step 축에서 얼마나 빨리 관측되는지가 달라진다.

### I.5 Dynamics of \(y\)

논문은 좌표 비율의 진화를 직접 적어, 왜 초기 최대 arm이 계속 최대 arm으로 남는지를 보여 준다. 시간은 결정을 바꾸는 것이 아니라 이미 존재하던 미세한 우위를 증폭한다는 점이 중요하다.

### I.6 Evolution of the collision term \(s_2\)

마지막 절은 \(s_2\)의 refined bound와 모멘트 관계를 이용해, good block polarization과 outer bad-mass decay가 실제 시간축에서 어떤 속도로 엮이는지를 더 세밀하게 분석한다.

## Appendix J Inner Dynamics of the Bad Arms

부록 J는 Appendix I의 bad-block 대응판이다.

### J.1 Within-bad composition and pushforward

bad block 내부 normalized composition \(z\)를 정의하면, logits에서 simplex로의 pushforward를 통해 독립적인 within-bad dynamics를 뽑아낼 수 있다.

### J.2 Bad-block drift: the same collision field with opposite sign

수학 구조는 good block과 거의 같지만, 부호가 반대다. 그래서 \(J>0\)에서는 good block이 sharpen될 때 bad block은 diffuse되고, \(J<0\)에서는 bad block 내부에서도 winner-take-all collapse가 일어난다.

### J.3 Internal-time form and direct correspondence with Section I

internal time을 적절히 잡으면 bad-block dynamics는 good-block dynamics의 time-reversed analogue로 볼 수 있다. 덕분에 good block에서 증명한 정리들을 부호 반전만으로 상당 부분 옮길 수 있다.

### J.4 Stability and global limits (bad-block counterpart of Section I.3)

<figure>
  <img src="/assets/images/Rate_or_Fate_RLVeR/rlver_table_03_stability_of_canonical_equilibria.png" alt="Table 3. Stability of canonical equilibria." />
  <figcaption><strong>Table 3.</strong> Stability of canonical equilibria.</figcaption>
</figure>

Table 3은 Table 2의 정확한 부호 반전판이다. \(J>0\)이면 bad block의 uniform equilibrium이 안정하고 vertex는 불안정하다. 즉, 남아 있는 bad mass는 한 잘못된 mode로 몰리지 않고 퍼진다. 반대로 \(J<0\)이면 uniform은 불안정하고 vertex가 안정해져, bad mode 내부에서도 collapse가 발생한다. anti-learning 구간에서 시스템이 왜 특정한 잘못된 해법에 집착하게 되는지가 여기서 설명된다.

Theorem J.6은 interior initialization 아래에서 \(J>0\)이면 bad block이 uniform point로, \(J<0\)이면 bad vertex로 간다는 전역 극한을 정리한다.

## Appendix K Shahshahani geometry and the within–good flow

부록 K는 good block dynamics를 다시 Shahshahani 기하의 언어로 해석한다.

### Shahshahani (Fisher) metric on the simplex.

within-good flow는 Shahshahani metric 아래에서 특정 potential의 gradient ascent로 쓸 수 있다. 따라서 good mode selection은 단순히 “확률이 우연히 몰리는 현상”이 아니라, 정보기하적으로 가장 자연스러운 ascent다.

### Interpretation (Herfindahl ascent in Fisher units).

그 potential은 사실상 Herfindahl-Hirschman concentration index와 연결된다. \(J>0\)일 때 good block이 한 mode로 집중되는 현상은, Fisher/Shahshahani 단위에서 concentration을 가장 빠르게 늘리는 방향으로 움직인 결과다. 즉, winner-take-all은 수치적 불안정성이 아니라 기하학적 최적화 방향의 귀결이다.

## Appendix L Hyperparameters and Training Details

<figure>
  <img src="/assets/images/Rate_or_Fate_RLVeR/rlver_table_04_training_configuration.png" alt="Table 4. Training configuration." />
  <figcaption><strong>Table 4.</strong> Training configuration.</figcaption>
</figure>

부록 L은 실험 전부에 사용한 하이퍼파라미터를 정리한다. 기본 모델은 Qwen2.5-3B이고, global batch size는 16, 총 학습 step은 1,410, 총 epoch은 2다. 프롬프트당 rollout은 8개이며, 학습 시 temperature는 1.0, top-p는 1.0으로 둔다. prompt와 response의 최대 길이는 각각 4000 token이다.

actor 학습은 PPO mini batch size 32, GRPO advantage estimation, clipping coefficient 0.2, Adam optimizer, weight decay 0.1, gradient norm clipping 1.0, constant scheduler, warmup 10 step을 사용한다. KL coefficient는 0.0으로 두어 reward-driven dynamics를 고립시킨다. 평가 시에는 greedy decoding(temperature 0.0)을 사용한다.

## Appendix M Noise Injection Pseudocode

[원문 Algorithm 1 삽입]

부록 M의 Algorithm 1은 noisy verifier wrapper를 정의한다. 구현은 단순하다. 먼저 oracle checker로 true correctness를 얻고, 그 결과가 정답이면 \(\delta_{\mathrm{FN}}\) 확률로 뒤집고, 오답이면 \(\delta_{\mathrm{FP}}\) 확률로 뒤집는다. 그러면 실험에서 사용한 operational reward가 만들어진다.

이 pseudocode는 논문 전체에서 사용한 noise model이 단지 이론적 장난이 아니라, 실제 학습 루프에 아주 직접적으로 삽입 가능한 wrapper라는 점을 보여 준다.

## Appendix N Data Sample

[원문 Appendix N Data Sample 삽입]

부록 N은 `kMarsh`라는 실제 예시 코딩 문제를 실어 둔다. 직사각형 격자에서 marsh를 피해 만들 수 있는 가장 큰 직사각형 울타리의 둘레를 구하는 문제로, 자연어 설명, 예시 격자, 함수 설명, 입력/출력 형식이 함께 주어진다. 논문은 이를 통해 실험 데이터가 어떤 종류의 programmatically verifiable coding task인지 구체적으로 보여 준다.

## 추가 설명

이 논문을 가장 정확하게 이해하는 방법은 “보상의 정확도” 대신 “보상의 방향성”을 보는 것이다. verifier가 오차를 가지더라도, 그 오차가 정답 쪽을 평균적으로 더 자주 밀어 준다면 \(J>0\)이고 학습은 성립한다. 하지만 verifier가 오답을 더 자주 밀어 주는 순간 \(J<0\)가 되어, 학습 루프 전체가 잘못된 objective를 증폭하는 기계로 바뀐다. 논문이 noisy reward를 단순한 품질 저하가 아니라 phase transition의 문제로 보는 이유가 여기에 있다.

또 하나 중요한 점은, 논문이 성능을 직접 추적하지 않고 bad mass \(p\)를 추적한다는 사실이다. accuracy를 직접 ODE로 닫기보다, “잘못된 모드에 얼마나 많은 질량이 남아 있는가”를 상태변수로 삼으면 식이 놀랄 만큼 간단해진다. 그리고 그 식의 부호가 \(J\) 하나로 정리된다. 즉, 이 논문의 이론적 힘은 거대한 LLM 학습 문제를 low-dimensional population dynamics로 바꾸는 데서 나온다.

\([p(1-p)]^2\)라는 prefactor도 매우 중요하다. 이 항은 왜 중간 난이도 프롬프트가 가장 잘 배우는지를 설명한다. 모델이 거의 항상 맞히는 문제는 더 좋아질 여지가 적고, 거의 항상 틀리는 문제는 증폭할 정답 signal이 적다. 그래서 RLVR는 본질적으로 “조금은 할 줄 아는 것”을 가장 빠르게 증폭한다. 이는 RLVR가 완전히 새로운 능력을 발명하기보다는, base model 내부에 이미 있는 약한 정답 모드를 키우는 데 강하다는 뜻이기도 하다.

false positive가 false negative보다 더 위험할 수 있다는 실험 결과도 이 이론과 잘 맞는다. false negative는 좋은 답을 놓치게 만들어 학습을 늦추지만, false positive는 나쁜 답을 좋은 답으로 적극 보상한다. 전자는 증폭 속도를 깎는 문제이고, 후자는 증폭 방향 자체를 어지럽히는 문제다. 특히 coding RLVR처럼 정답 프로그램의 표현이 다양할 때는 false positive가 잘못된 스타일이나 얕은 heuristic을 강화해 mode collapse를 유도하기 쉽다.

good block과 bad block 내부 dynamics를 따로 본 것도 이 논문의 큰 장점이다. 전체적으로는 bad mass가 줄어드는지 늘어나는지가 중요하지만, 동시에 good block 내부에서는 어떤 정답 모드 하나가 지배적이 되고, bad block 내부에서는 질량이 퍼지거나 반대로 하나의 잘못된 모드로 몰린다. 그래서 RLVR는 단순히 “정확도가 오른다/내린다”의 문제가 아니라, 어떤 reasoning diversity를 남기고 어떤 다양성을 없애느냐의 문제이기도 하다.

KL regularization에 대한 논문의 해석도 실용적이다. 많은 경우 KL은 학습을 더 안전하게 만드는 규제항으로 쓰이지만, 이 논문은 KL이 anti-informative verifier를 informative하게 바꾸지는 못한다고 분명히 말한다. KL이 할 수 있는 것은 boundary collapse를 interior equilibrium으로 바꾸어 극단화를 늦추는 것뿐이다. 즉, KL은 안정성 장치이지 신호 복원 장치가 아니다.

마지막으로, 이 논문은 noisy RLVR를 다룰 때 무엇을 먼저 진단해야 하는지도 사실상 제시한다. 제일 먼저 봐야 할 것은 verifier의 순 판별력 \(J\)다. 그 다음에야 rollout 수, training step, KL 강도, clipping, sampling variance 같은 것들이 의미를 갖는다. \(J>0\)라면 더 많은 계산과 더 나은 geometry control이 도움이 될 수 있다. 하지만 \(J\le 0\)라면 그 어떤 계산도 근본 문제를 해결하지 못한다. 이 점에서 RLVεR는 단순한 이론 논문이 아니라, noisy verifier 아래에서 RLVR를 설계하고 진단하기 위한 기본 좌표계를 제공하는 논문이라고 볼 수 있다.
