    ---
    title: "Rate or Fate? RLVεR: Reinforcement Learning with Verifiable Noisy Rewards"
    math: true
    ---

    <script>
window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true
  },
  svg: { fontCache: 'global' }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

> 편집 원칙  
> - 수식은 MathJax 기준으로 다시 정리했습니다.  
> - Figure/Table은 실제 이미지를 직접 삽입하지 않고, **어떤 원문 도판을 넣어야 하는지**를 안내하는 placeholder로 통일했습니다.  
> - 이해 점검 질문, 스터디 체크리스트, 로드맵 등 부속 학습 패키지는 제거해 **논문 해설 본문 중심**으로 재구성했습니다.  


## 전문가 관점 요약

RLVεR 논문은 최근 코드/추론 분야에서 주목받는 RLVR 계열 학습을 정면으로 분석한다. 핵심 메시지는 단순하다. **검증 가능한 보상(verifiable reward)이 있다고 해서 학습이 자동으로 안전하거나 안정적인 것은 아니며, verifier가 noisy하면 학습은 질적으로 다른 동역학을 보일 수 있다.** 저자들은 이를 intuition 수준이 아니라, coarse-grained bandit 모델과 replicator-style dynamics를 통해 분석한다.

이 논문의 가장 중요한 공헌은 noisy verifier를 TPR/FPR, 나아가 **Youden’s index \(J\)** 로 압축해 “학습이 가능한 구간과 불가능한 구간”을 나누는 임계 구조를 드러낸 점이다. 즉, 보상 노이즈는 언제나 단순한 성능 저하로 나타나는 것이 아니라, 특정 임계선을 넘으면 학습 방향 자체를 뒤집을 수 있다. 이 때문에 제목의 “Rate or Fate?”가 정확한 문제 제기다. 어떤 경우에는 속도만 늦어지지만, 어떤 경우에는 최종 수렴점 자체가 달라진다.

실무적으로 이 문헌이 중요한 이유는, 최근의 코딩/추론 RL이 자주 의존하는 pass/fail 검증기가 결코 무오류가 아니기 때문이다. 특히 false positive는 나쁜 해법에 보상을 부여해 질량을 잘못 증폭시키고, false negative는 좋은 해법을 충분히 강화하지 못하게 만든다. 논문은 이 효과가 group-normalized RL과 결합할 때 어떻게 증폭되거나 상쇄되는지를 구조적으로 보여 준다.

다만 이 논문을 읽을 때는 이상화 수준도 함께 봐야 한다. 저자들의 분석은 해석 가능한 축약 모델을 사용하므로, 실제 대규모 LLM 학습의 모든 이질성과 분포 이동을 반영하지는 않는다. 그럼에도 불구하고 연구적으로 매우 가치 있는 이유는, 복잡한 RLVR 현상 뒤에 있는 **일차 원리(first-order principle)** 를 제시했기 때문이다. 이후 noisy verifier 설계, verifier calibration, curriculum, difficulty-aware sampling 같은 후속 연구는 모두 이 문제의식 위에서 이해할 수 있다.


# Rate or Fate? RLVεR: Reinforcement Learning with Verifiable Noisy Rewards

## 확장 해설본(annotated expansion)

- **원문 제목**: *RLVεR: Reinforcement Learning with Verifiable Noisy Rewards*
- **저자**: Ali Rad, Khashayar Filom, Darioush Keivan, Peyman Mohajerin Esfahani, Ehsan Kamalinejad
- **출처**: arXiv:2601.04411v1, 2026-01-07
- **문서 표기 날짜**: 2026-01-09
- **코드 저장소**: `https://github.com/cognichip/Noisy-RL`
- **기준 문헌**: arXiv PDF 원문 전체(본문 1–8절, 부록 A–N)

---

## 작성 원칙

이 문서는 원문 논문의 전체 구조를 따라가며, 각 절의 논지를 빠짐없이 재구성하고 필요한 배경과 직관을 덧붙인 **확장 해설본**이다. 본문과 부록의 정의, 수식, 표, 그림 캡션, 실험 설정, 제한사항, 향후 과제를 모두 다루는 것을 목표로 한다. 원문에 직접 근거가 있는 내용은 가능한 한 섹션·식·그림·표·부록 번호를 함께 명시하였다. 원문에 없는 구현 세부나 실험 조건은 추정하지 않으며, 필요한 경우 **“논문에 명시 없음”**이라고 분리해 적는다.

표기 규칙은 다음과 같다.

- **[원문 근거]**: 논문 본문이나 부록에 직접적으로 근거가 있는 진술.
- **[추가 설명(일반 지식)]**: 독해를 돕기 위한 배경 설명이나 직관.
- **[추가 아이디어(원문 외)]**: 논문이 직접 주장하지는 않지만, 스터디·응용·후속 연구 관점에서 유의미한 확장 제안.

---

# Abstract 확장 해설

## A. 섹션 길잡이

Abstract의 핵심 질문은 다음 한 문장으로 요약된다. **검증 가능한 보상으로 수행하는 RLVR에서 검증기(verifier)가 noisy할 때, 그 노이즈는 학습을 단지 느리게 만드는가, 아니면 학습의 최종 귀결 자체를 뒤집는가?** (근거: Abstract)

이 질문을 이해하기 위해 필요한 선수 개념은 RLVR, GRPO, false positive/false negative, Youden’s index, replicator dynamics이다. Abstract를 읽을 때 특히 확인해야 할 것은 세 가지다. 첫째, 저자들이 왜 RLVR을 multi-armed bandit 관점으로 다시 쓰는지. 둘째, verifier noise가 왜 단일 스칼라 지표 \(J=\mathrm{TPR}-\mathrm{FPR}\)로 요약될 수 있는지. 셋째, “rate, not fate”라는 결론이 정확히 어떤 조건에서 성립하는지다. (근거: Abstract)

## B. 원문 내용 재구성

[원문 근거] 저자들은 RLVR을 “completion을 샘플링하고, verifier로 채점하고, 그 결과로 정책을 업데이트하는” 단순하지만 강력한 학습 패러다임으로 제시한다. 그러나 실제 verifier는 거의 항상 깨끗하지 않다. 코딩 과제의 unit test는 제한된 corner case만 검증하고, 인간 또는 합성 라벨은 불완전하며, LLM judge는 noisy하고 exploit되기 쉽다. 이런 문제는 난도가 높은 영역, 특히 코딩에서 더 심화된다. 테스트가 희소하고 점점 더 모델 생성 산출물에 의존하기 때문이다. (근거: Abstract)

[원문 근거] 이 문제의식을 바탕으로 저자들은 다음의 실용적 질문을 던진다. **verification noise는 학습 속도(rate)만 늦추는가, 아니면 학습의 운명(fate)까지 뒤집는가?** 이에 답하기 위해 RLVR dynamics를 해석 가능한 multi-armed bandit으로 coarse-grain하고, 이를 GRPO와 결합하여 수학적으로 다룬다. false positive와 false negative를 명시적으로 모델링하고, completion들을 반복적으로 나타나는 reasoning mode로 묶으면, 확률단체(probability simplex) 위에서 replicator-style, 즉 자연선택형 동역학이 도출된다는 것이 Abstract의 요지다. (근거: Abstract)

[원문 근거] 도출된 흐름은 두 층으로 분리된다. 하나는 correct mode 내부 경쟁이고, 다른 하나는 incorrect mode 전체 질량의 1차원 진화다. 특히 incorrect mode 총질량의 drift는 오직

$$
J = \mathrm{TPR}-\mathrm{FPR}
$$

에 의해 결정된다고 요약된다. 이로부터 sharp phase transition이 도출된다. \(J>0\)이면 incorrect mode의 질량이 소멸 방향으로 밀려 learning이 일어나고, \(J=0\)이면 시스템은 neutral하며, \(J<0\)이면 incorrect mode가 증폭되어 anti-learning과 collapse가 발생한다. 또한 \(J>0\)인 경우 noise는 주로 수렴 시간만 재조정하며, 최종 basin은 바꾸지 않는다는 결론이 제시된다. 저자들은 Python programming task에 synthetic noise를 주입한 실험으로 이러한 \(J=0\) 경계를 확인했다고 밝힌다. (근거: Abstract)

## C. 친절한 직관과 해설

[추가 설명(일반 지식)] Abstract의 가장 중요한 압축은 verifier noise를 \(J=\mathrm{TPR}-\mathrm{FPR}\) 하나로 보는 관점이다. \(\mathrm{TPR}\)은 맞는 답을 맞다고 인정하는 확률이고, \(\mathrm{FPR}\)은 틀린 답을 맞다고 잘못 인정하는 확률이다. 따라서 \(J\)는 “검증기의 칭찬이 실제 정답성과 같은 방향인가”를 나타내는 부호로 읽을 수 있다. \(J>0\)이면 verifier의 칭찬은 평균적으로 정답 쪽을 더 많이 가리키므로 학습 방향이 유지된다. 반대로 \(J<0\)이면 verifier가 틀린 답을 더 자주 보상하므로, RL 업데이트는 체계적으로 잘못된 방향으로 작동한다. (근거: Abstract)

---

# 1. Introduction 확장 해설

## A. 섹션 길잡이

서론의 목적은 세 가지다. 첫째, 왜 RLVR과 GRPO가 최근 LLM 학습에서 핵심적 위치를 차지하게 되었는지 설명한다. 둘째, group-normalized RL이 실전에서 의존하는 sequence-level reward가 얼마나 취약한지 문제를 제기한다. 셋째, noisy verifier 문제를 false positive/false negative와 Youden’s index \(J\)라는 최소 모델로 정식화한다. (근거: Section 1)

이 절을 읽을 때는 다음 질문들을 계속 염두에 두는 것이 좋다. verifier가 완벽하지 않을 때 RLVR은 어디까지 견딜 수 있는가? 단순히 noisy label의 평균 효과를 보는 것이 아니라, 어떤 **임계선**이 존재하는가? 그리고 코딩 과제는 왜 다른 검증형 작업보다 더 예민한가? (근거: Section 1)

## B. 원문 내용 재구성

### 1.1 Reinforcement Learning과 LLM의 최근 맥락

[원문 근거] 저자들은 최근 LLM의 reasoning 역량 향상에서 RL, 특히 RLVR과 GRPO 같은 group-normalized 알고리즘이 큰 역할을 했다고 서술한다. 이러한 계열의 방법론은 self-play, 환경 상호작용, reward-based feedback을 통해 창의성과 지능이 출현할 수 있다는 오래된 가설에 새로운 힘을 부여했다. (근거: Section 1, “Reinforcement Learning and LLMs”)

### 1.2 Group-normalized RL의 장점과 근본적 의존성

[원문 근거] GRPO와 같은 group-normalized 접근은 수학이나 코드 생성처럼 verifier가 존재하는 영역에서 critic 또는 별도 reward model 없이도 sequence-level advantage를 근사할 수 있다는 장점을 가진다. 적은 수의 rollout만으로도 동일 프롬프트에서 생성된 completion들을 상대 비교하여 업데이트할 수 있다는 점이 강조된다. 그러나 저자들은 동시에 중요한 사실을 분명히 한다. **이 모든 알고리즘의 토대는 여전히 sequence-level reward**라는 것이다. 따라서 reward의 품질이 낮다면, RL training 전체가 본질적으로 영향을 받는다. (근거: Section 1, “Group-Normalized RL and RLVR”)

[원문 근거] 이 지점에서 서론은 일련의 질문을 제기한다. RL은 ground truth label과 reward의 품질에 얼마나 민감한가? 성능은 robust한가? 혹은 단순히 “garbage in, garbage out”처럼 noisy reward에 비례해 degraded되는가? 이 질문은 이후 논문 전체의 출발점이 된다. (근거: Section 1)

### 1.3 라벨 프리 혹은 AI feedback 접근의 취약성

[원문 근거] 저자들은 LLM-as-Judge, RLAIF와 같이 인간 선호 라벨을 AI feedback으로 대체하려는 접근, 그리고 self-rewarding, label-free, majority voting, reasoning trace consistency, self-certainty와 같은 대안들이 존재함을 언급한다. 그러나 이들 역시 동일한 false positive/false negative 문제로부터 자유롭지 않다고 지적한다. 즉, supervision source를 사람에서 모델로 바꾸거나, 외부 정답을 내부 신호로 대체하더라도 reward noise의 구조적 문제는 그대로 남는다. (근거: Section 1)

### 1.4 Motivation: 정답이 직접 주어지지 않아도 학습은 가능한가?

[원문 근거] 저자들은 RL이 reward와 환경 피드백에 본질적으로 의존하므로, 피드백 품질에 매우 민감할 수밖에 없다고 본다. 따라서 “피드백이 noisy하고 직접적인 ground truth가 없는 상황에서, 에이전트는 여전히 학습하고 self-improve할 수 있는가?”라는 질문이 자연스럽게 제기된다. 이 질문을 정밀하게 다루기 위해, 본 논문은 noisy feedback 하의 RL dynamics를 분석하겠다고 선언한다. (근거: Section 1, “Motivation”)

### 1.5 코딩 RLVR가 특히 문제적인 이유

[원문 근거] 논문은 코딩 과제에서 verifier noise가 특히 심각하다고 주장한다. unit test는 본질적으로 불완전하며, edge case를 놓칠 수 있고, 기능적으로 올바른 여러 구현 중 일부만 통과시킬 수 있다. 수학 단답형 문제나 객관식 과제처럼 답이 고정된 경우와 달리, 코드는 의미적으로 동등한 정답이 여러 개 존재한다. 또한 난도가 증가할수록 test coverage와 fidelity는 악화될 가능성이 크며, 극단적인 경우 pass/fail이 실제 correctness와 거의 무상관해질 수 있다. 저자들이 실험을 Python programming task에 집중한 이유가 바로 여기에 있다. (근거: Section 1, “Noisy Reward for Coding Tasks”)

### 1.6 False Positive와 False Negative의 최소 모델

[원문 근거] 실전에서 supervision noise는 주로 두 방식으로 나타난다. 첫째는 **false positive**다. 즉, 잘못된 해답이 positive reward를 받는 경우다. 둘째는 **false negative**다. 즉, 올바른 해답이 reward를 받지 못하거나 오히려 불이익을 받는 경우다. 저자들은 이러한 노이즈를 프롬프트별 sequence-level grader의 noisy binary reward \(r\in\{0,1\}\)로 표현하고,

$$
\delta_{FN}=\Pr(r=0\mid good), \qquad \delta_{FP}=\Pr(r=1\mid bad)
\tag{1}
$$

로 정의한다. 이 에러율은 일반적으로 시간의존적일 수도 있다고 명시한다. 즉, \(\delta_{FN}(t), \delta_{FP}(t)\)처럼 training 과정에서 바뀔 수 있다. (근거: Section 1, Eq. 1)

### 1.7 Youden’s index \(J\)의 도입과 해석

[원문 근거] 저자들은 위 두 error rate를 하나의 스칼라로 요약한다.

$$
J := 1-\delta_{FN}-\delta_{FP}=\mathrm{TPR}-\mathrm{FPR} \in [-1,1].
\tag{2}
$$

이는 통계적 결정이론에서 Youden’s index로 알려져 있으며, verifier의 순수한 판별력을 요약한다. 논문은 세 가지 해석을 직접 제시한다.

- \(J=1\): perfect rewarder, 즉 \(\mathrm{TPR}=1\), \(\mathrm{FPR}=0\)
- \(J=0\): chance-level rewarder, 즉 \(\mathrm{TPR}=\mathrm{FPR}\)
- \(J<0\): inverted 또는 anti-informative rewarder

또한 ROC curve 관점에서 \(J\)는 verifier의 ROC가 random diagonal에서 얼마나 위에 있는지를 나타내는 수직 거리라고 설명한다. (근거: Section 1, Eq. 2)

### 1.8 논문이 제기하는 핵심 연구 질문

[원문 근거] 서론 후반에서 저자들은 본 논문이 묻는 질문을 네 항목으로 명시한다.

1. RLVR은 어느 정도의 reward sloppiness까지 견딜 수 있는가?
2. noisy reward 하에서 RLVR의 learning dynamics는 어떻게 기술되는가?
3. noise가 언제 rate만 바꾸고, 언제 fate를 바꾸는가?
4. verifier noise는 어떠한 임계점에서 phase transition을 일으키는가?

즉, 논문의 문제의식은 단순히 “노이즈가 성능을 얼마나 깎는가”가 아니라, **학습가능성과 붕괴 사이의 경계가 어디인가**에 놓여 있다. (근거: Section 1, “THIS PAPER ASKS”)

### 1.9 RLVεR 프레임워크의 선언

[원문 근거] 이러한 질문에 답하기 위해 저자들은 RLVεR, 즉 *Reinforcement Learning with Verifiable Noisy Rewards* 프레임워크를 제안한다고 말한다. 여기서 핵심 제어 손잡이는 바로 \(J\)이며, 이후 본문은 noisy verifier의 sloppiness가 GRPO dynamics를 어떻게 바꾸는지 이론과 실험 양쪽에서 추적한다. (근거: Section 1 말미)

## C. 친절한 직관과 해설

[추가 설명(일반 지식)] 서론의 중요한 통찰은, noisy supervision의 문제가 단순한 “annotation quality 저하” 문제가 아니라는 점이다. RL에서는 보상이 곧 최적화 방향이다. 따라서 false negative는 유용한 신호를 약화시키는 효과가 있지만, false positive는 아예 **잘못된 방향으로 gradient를 밀어버릴 수 있다**. 이 논문은 그 차이를 \(J\)의 부호 변화라는 형태로 압축해 보여준다. 코딩 과제에서 unit test가 불완전하면, 모델은 실제로 더 좋은 코드를 만들기보다 “테스트를 통과하는 방식”을 학습할 수 있다. 서론은 바로 이 위험을 정면으로 다룬다. (근거: Section 1)

---
# 2. RLVR with Sloppy Rewards 확장 해설

## A. 섹션 길잡이

이 절의 목적은 실제 GRPO/RLVR 업데이트를 최소한의 수식으로 적고, 그 결과 잘못된 해법 질량이 어떻게 이동하는지를 연속시간 방정식으로 도출하는 데 있다. 핵심 질문은 다음과 같다. 동일 프롬프트에서 여러 completion을 샘플링하고, 이를 group-normalized reward로 비교해 업데이트할 때, 그 업데이트는 “좋은 답”과 “나쁜 답”의 확률질량을 어떤 방향으로 이동시키는가? (근거: Section 2)

특히 확인해야 할 지점은 세 가지다. 첫째, GRPO의 1-step이 sampling–scoring–group normalization–policy gradient의 네 단계로 정리된다는 점. 둘째, good/bad 이분법 아래 bad mass \(p\)의 1차원 ODE가 나온다는 점. 셋째, noisy reward를 넣으면 advantage gap이 \(J/\sigma(p)\) 꼴로 정리된다는 점이다. (근거: Section 2)

## B. 원문 내용 재구성

### 2.0 GRPO / group-normalized RLVR의 한 iteration

[원문 근거] 저자들은 한 프롬프트 \(x\)에 대해, 현재 정책 \(\pi_\theta(\cdot\mid x)\)로 \(G\)개의 독립 completion을 샘플링하는 것으로 시작한다.

$$
y_1,\dots,y_G \sim \pi_\theta(\cdot\mid x).
$$

동치로, softmax logits에서 categorical index를 샘플링한다고 볼 수도 있다. 각 completion에는 programmatic rule 또는 learned reward model을 이용해 raw reward가 부여된다.

$$
r_g := r(x,y_g).
$$

그 다음 각 reward는 현재 group 내부에서 표준화되어 per-sample advantage로 변환된다.

$$
A_g^b = \frac{r_g-\bar r}{\sigma_r+\varepsilon},
$$

여기서 \(\bar r\)과 \(\sigma_r\)는 해당 batch 혹은 group의 empirical mean과 standard deviation이다. 마지막으로 정책은 다음과 같이 업데이트된다.

$$
\Delta\theta
= \eta \frac{1}{G}\sum_{g=1}^{G} A_g^b \,\nabla_\theta \log \pi_\theta(y_g\mid x).
\tag{3}
$$

이 식이 Section 2 전개의 출발점이다. 저자들은 PPO-style clipping이나 KL penalty는 이후 §6에서 다루겠다고 명시한다. (근거: Section 2, Eq. 3)

### 2.1 가장 단순한 two-class setup: good vs bad

[원문 근거] 업데이트 메커니즘의 본질을 보기 위해 저자들은 먼저 binary outcome setup을 고려한다. 모델은 “good” 혹은 “bad” 해답을 내며, bad solution을 생성할 확률을 \(p\)라 두고 이를 logit \(z\)로 parameterize한다.

$$
p = \sigma(z)=\frac{1}{1+e^{-z}} \equiv \pi(\mathrm{Bad}).
$$

per-sample normalized advantage \(A^b\)에 대해 expected logit update는 score function과의 상관으로 주어진다.

$$
\Delta z \propto \mathbb{E}\big[A^b\nabla_z\log \pi(a)\big].
$$

이진의 경우 score function은 단순화된다.

$$
\nabla_z \log \pi(\mathrm{bad}) = 1-p,
\qquad
\nabla_z \log \pi(\mathrm{good}) = -p.
$$

조건부 기대 advantage를

$$
f(\mathrm{bad})=\mathbb{E}[A^b\mid \mathrm{bad}],
\qquad
f(\mathrm{good})=\mathbb{E}[A^b\mid \mathrm{good}]
$$

로 두면,

$$
\mathbb{E}[A^b\nabla_z\log \pi(a)]
= p(1-p)\big(f(\mathrm{bad})-f(\mathrm{good})\big)
$$

을 얻는다. 저자들은 이를 continuous-time으로 넘겨

$$
\dot p(t)
=
-\eta\,[p(t)(1-p(t))]^2\,\big(f(\mathrm{good})-f(\mathrm{bad})\big)
\tag{4}
$$

라는 “GRPO dynamics”를 제시한다. 해석은 명확하다. \(f(\mathrm{good})>f(\mathrm{bad})\)이면 \(\dot p<0\)이므로 bad mass가 줄고, 정확도 \((1-p)\)는 올라간다. (근거: Section 2.1, Eq. 4)

### 2.1.1 Replicator dynamics와의 연결

[원문 근거] 논문은 위 식이 보다 일반적으로 replicator dynamics의 한 예라고 설명한다.

$$
\dot p_i(t)=p_i(t)\big(f_i(p(t)) - \bar f(p(t))\big),
\qquad
\bar f(p)=\sum_j p_j f_j(p).
$$

즉 평균 fitness보다 높은 타입은 확률이 증가하고, 낮은 타입은 감소한다. 저자들은 GRPO를 “자연선택과 유사한 흐름”으로 해석하며, 이것이 후속 절들에서 확률단체(simplex) 기하와 자연스럽게 만난다고 본다. (근거: Section 2.1)

### 2.2 Noisy Rewards의 삽입

[원문 근거] 이제 reward가 noisy binary signal \(r\in\{0,1\}\)이고, Section 1에서 정의한 false negative 및 false positive가 존재한다고 가정한다. bad mass가 \(p\)일 때 기대 reward는 다음과 같이 계산된다.

$$
q(p):=\mathbb{E}[r]
=(1-\delta_{FN})-(1-\delta_{FP}-\delta_{FN})p
=(1-\delta_{FN})-Jp.
$$

reward가 Bernoulli이므로 분산은

$$
\sigma(p)^2 = \mathrm{Var}[r]=q(p)(1-q(p))
$$

이다. 여기서 중요한 결과는 group normalization 아래에서 good과 bad 사이 conditional expected advantage의 차이가 다음과 같이 깔끔하게 정리된다는 사실이다.

$$
\mathbb{E}[A^b\mid \mathrm{good}] - \mathbb{E}[A^b\mid \mathrm{bad}]
= \frac{1-\delta_{FN}-\delta_{FP}}{\sigma(p)}
= \frac{J}{\sigma(p)}.
\tag{5}
$$

즉 noisy reward의 복잡한 효과가 결국 분자 \(J\), 분모 \(\sigma(p)\)인 단순한 꼴로 응축된다. (근거: Section 2.2, Eq. 5)

[원문 근거] 따라서 \(J\)는 단순한 보조지표가 아니라 learning direction의 부호를 결정하는 핵심량이 된다. \(J>0\)이면 verifier는 평균적으로 informative하므로 업데이트는 정답 쪽을 향하고, \(J<0\)이면 verifier는 anti-informative하여 업데이트가 반대로 작동한다. \(J=0\)이면 gradient signal은 사라지고, 표본 잡음에 의한 중립적 drift만 남는다. (근거: Section 2.2)

## C. 친절한 직관과 해설

[추가 설명(일반 지식)] Section 2의 가장 중요한 수식은 식 (5)다. 표면적으로는 noisy reward가 여러 형태로 들어오는 것처럼 보이지만, group normalization까지 거친 뒤 good과 bad 사이 평균 advantage 차이는 결국 \(J/\sigma(p)\) 하나로 정리된다. 이것은 “정확히 어떤 방향으로 학습되는가”를 결정하는 정보가 mostly \(J\)의 부호에 들어 있다는 뜻이다. 이 관점은 이후 3절에서 상전이, 5절에서 simplex geometry, 6절에서 clipping과 KL regularization을 설명하는 기반이 된다. (근거: Section 2)

---

# 3. Phase Transition 확장 해설

## A. 섹션 길잡이

Section 3은 논문의 핵심 결과를 가장 직접적으로 제시하는 절이다. 앞 절에서 얻은 bad mass ODE에 noisy reward model을 대입해, 학습이 세 가지 regime—learning, neutral, anti-learning—으로 나뉜다는 사실을 보인다. 또한 \(J>0\)일 때 noise가 주로 수렴 속도만 바꾸는 이유, 그리고 왜 중간 난이도의 프롬프트가 가장 잘 학습되는지를 정리한다. (근거: Section 3)

이 절에서 확인해야 할 포인트는 네 가지다. 첫째, \(J\)의 부호에 따라 \(p(t)\)의 drift 부호가 완전히 결정된다. 둘째, \(J=0\)은 단순한 성능 저하가 아니라 **phase boundary**다. 셋째, 완전 오라클에서는 \(t^{-2}\), 일반 noisy case에서는 주로 \(t^{-1}\) tail이 나타난다. 넷째, learnability는 \(p\approx 1/2\)의 중간 난이도에서 가장 크다. (근거: Section 3)

## B. 원문 내용 재구성

### 3.0 noisy reward가 들어간 bad mass ODE

[원문 근거] 식 (4)와 식 (5)를 결합하면, bad mass dynamics는 다음의 1차원 ODE로 정리된다.

$$
\dot p
=
-\eta\,\frac{J}{\sigma(p)}\,p^2(1-p)^2,
\qquad
J=\mathrm{TPR}-\mathrm{FPR}.
\tag{6}
$$

여기서 \(p\)는 bad mode 총질량이며, \(\sigma(p)\)는 group-normalized reward의 상태의존 표준편차다. 저자들은 \(\mathrm{TPR}\)과 \(\mathrm{FPR}\)이 정책과 verifier가 함께 변할 경우 시간의존적일 수 있지만, 식 (6)은 주어진 시점의 국소적 학습 방향을 정확히 포착한다고 설명한다. (근거: Section 3, Eq. 6)

### 3.0.1 세 가지 학습 레짐

[원문 근거] 식 (6)에서 \(\sigma(p)>0\)라고 두면, drift의 부호는 전적으로 \(J\)의 부호에 의해 정해진다.

- \(J>0\)이면 \(\dot p<0\): bad mass 감소, 즉 **learning**
- \(J=0\)이면 \(\dot p=0\): 방향성 없는 **neutral regime**
- \(J<0\)이면 \(\dot p>0\): bad mass 증가, 즉 **anti-learning**

저자들은 이 결과를 sharp phase transition이라고 부른다. 임계 경계는 정확히 \(\mathrm{TPR}=\mathrm{FPR}\), 즉 \(J=0\)이다. (근거: Section 3)

### 3.0.2 perfect oracle의 기준 동역학

[원문 근거] noise가 전혀 없고 verifier가 완전하다면 \(J=1\)이며, bad mass ODE는

$$
\dot p = -\eta\,p^{3/2}(1-p)^{3/2}
\tag{7}
$$

로 단순화된다. 이 식은 noise-free learning의 기준 동역학으로 사용된다. (근거: Section 3, Eq. 7)

### 3.1 임계점의 분기와 고정점 구조

[원문 근거] \(J\neq 0\)일 때 식 (6)은 두 개의 경계 고정점 \(p^\star=0\)과 \(p^\star=1\)을 가진다. 어느 쪽이 attractor인지 여부는 \(\mathrm{sign}(J)\)가 결정한다.

- \(J>0\): \(p=0\)이 전역 attractor, \(p=1\)은 repeller
- \(J<0\): \(p=1\)이 전역 attractor, \(p=0\)은 repeller
- \(J=0\): ODE가 항등적으로 0이 되어 \([0,1]\) 전체가 neutrally stable fixed-point continuum

즉 \(J=0\)에서는 특정 점 하나가 아니라 전체 구간이 고정점 집합이 된다. (근거: Section 3.1)

### 3.1.1 \(J=1\)의 닫힌 형태 해와 support barrier

[원문 근거] 완전 오라클 \(J=1\)의 경우 저자들은 bad mass의 명시적 해를 제시한다.

$$
p(t)=
\begin{cases}
\displaystyle
\frac12+\frac12\cdot
\frac{\varphi(p_0)-\frac{\eta}{2}t}{\sqrt{4+\left(\varphi(p_0)-\frac{\eta}{2}t\right)^2}},
& p_0\neq 0,1,\\[10pt]
0,1,& p_0=0,1,
\end{cases}
\qquad
\varphi(p)=\frac{2p-1}{\sqrt{p(1-p)}}.
\tag{8}
$$

또한 \(p_0\neq 0,1\)라면 late time에서

$$
p(t)\sim \frac{4}{\eta^2 t^2}\to 0
$$

이므로 정확도 \(1-p(t)\)는 \(t^{-2}\) 꼬리로 1에 접근한다. (근거: Section 3.1.1, Eq. 8)

[원문 근거] 이 해에서 저자들이 특히 강조하는 것은 **support barrier**다. 경계 상태 \(p_0\in\{0,1\}\)는 absorbing하다. 즉 초기 정책이 특정 모드에 정확히 0의 확률을 부여하면, RLVR은 그 모드를 새로 생성해낼 수 없다. 특히 \(p_0=1\), 즉 정답 모드가 전혀 샘플링되지 않으면 RLVR은 그 프롬프트에서 “이륙”하지 못한다. 저자들은 이 점을 들어 RLVR이 새로운 capability를 무(無)에서 창조하기보다는, 이미 존재하는 reasoning path를 증폭·재가중하는 메커니즘에 가깝다고 해석한다. 또한 pass@1은 오를 수 있지만 large-k coverage는 줄 수 있다는 기존 관찰과도 정합적이라고 지적한다. (근거: Section 3.1.1)

### 3.1.2 점근 꼬리: \(t^{-1}\)과 \(t^{-2}\)

[원문 근거] 저자들은 late-time behavior가 attractor 근처에서 reward variance가 0으로 가는지 여부에 달려 있다고 분석한다. 경계 분산을

$$
\sigma_0=\sqrt{(1-\delta_{FN})\delta_{FN}},
\qquad
\sigma_1=\sqrt{\delta_{FP}(1-\delta_{FP})}
$$

로 두면, \(J>0\)의 attractor \(p=0\) 근처에서 두 경우가 갈린다.

1. **비퇴화 노이즈** \((\delta_{FN}>0)\): \(\sigma(p)\to \sigma_0>0\) 이므로

$$
\dot p = -\eta\frac{J}{\sigma_0}p^2 + o(p^2)
\quad\Rightarrow\quad
p(t)\sim \frac{\sigma_0}{\eta J}\frac{1}{t}.
$$

2. **variance-degenerate case** \((\delta_{FN}=0)\): \(\sigma(p)\sim\sqrt{Jp}\) 이므로

$$
\dot p = -\eta\sqrt{J}\,p^{3/2}+o(p^{3/2})
\quad\Rightarrow\quad
p(t)\sim \frac{4}{\eta^2 J}\frac{1}{t^2}.
$$

반대로 \(J<0\)인 경우 attractor는 \(p=1\)이며, \(u(t)=1-p(t)\)로 두면 일반적으로

$$
u(t)=1-p(t)\sim \frac{\sigma_1}{\eta |J|}\frac{1}{t}
$$

의 tail을 얻는다. 저자들은 이로부터 error decay의 보편 패턴을 다음과 같이 요약한다.

- attractor에서 variance가 nonzero이면 \(O(t^{-1})\)
- \(\delta_{FN}=0\), \(J>0\)의 degenerate case이면 \(O(t^{-2})\)

(근거: Section 3.1.2)

### 3.2 Rate, not Fate

[원문 근거] 저자들은 \(J>0\)인 한 noisy reward와 noise-free reward가 같은 basin으로 수렴한다고 주장한다. 즉 1차원 ODE 수준에서는 attractor가 동일하므로, noise는 최종 도달점이 아니라 시간척도를 바꾸는 역할을 한다. 이를 식으로 요약하면 noisy와 perfect dynamics 사이의 상대 속도는 대체로

$$
\frac{\dot p_{\mathrm{noisy}}}{\dot p_{\mathrm{perfect}}}
\propto \frac{1}{J}
\tag{9}
$$

꼴로 재스케일된다. 예컨대 \(J=0.5\)라면, 동일한 궤적을 따라가기 위해 대략 두 배의 compute step이 필요하다는 식의 해석이 가능하다. 이것이 논문의 표어인 “rate, not fate”의 수학적 표현이다. 물론 이 명제는 \(J>0\)에서만 성립한다. \(J\le 0\)이면 애초에 basin 구조 자체가 달라지므로 fate가 바뀐다. (근거: Section 3.2, Eq. 9)

### 3.3 Maximal Learnability at Intermediate Bad Mass

[원문 근거] noiseless case \((J=1)\)에서 bad mass의 instantaneous improvement 크기는

$$
|\Delta p| \propto [p(1-p)]^{3/2}
$$

에 비례한다. 따라서 prefactor \(p(1-p)\)는 \(p=1/2\)에서 최대가 되고, 저자들은 이를 “가장 잘 배우는 구간은 중간 bad mass”라고 해석한다. 이미 거의 다 맞는 프롬프트 \((p\approx 0)\)는 더 개선할 여지가 적고, 반대로 거의 항상 틀리는 프롬프트 \((p\approx 1)\)는 정답 signal을 샘플링하기 어려워 증폭할 씨앗이 부족하다. (근거: Section 3.3)

[원문 근거] 저자들은 이 결과를 기존의 learnability 관찰과 연결한다. progress bound가 \(p(1-p)\) 꼴로 나타나거나, Bernoulli reward variance가 \(q(1-q)\)로 \(q=1/2\)에서 최대가 되는 기존 연구들과 본 논문의 mean-field GRPO dynamics가 정합적이라는 것이다. 또한 비대칭 노이즈가 존재할 때도 최적 learnability 지점은 이동할 수 있지만, 여전히 “중간 정도 bad mass”의 영역에 위치한다고 설명한다. (근거: Section 3.3)

## C. 친절한 직관과 해설

[추가 설명(일반 지식)] Section 3의 가장 중요한 메시지는, verifier quality가 충분히 나쁘면 학습이 단지 느려지는 것이 아니라 **방향을 바꾼다**는 점이다. supervised learning에서 noisy label은 종종 성능을 떨어뜨리는 정도로 이해되지만, RL에서는 reward가 곧 optimization direction이므로, false positive가 심해지면 policy는 실제로 더 나쁜 해답군을 선호하는 방향으로 이동할 수 있다. 이 논문은 바로 그 경계가 \(J=0\)임을 보여준다. 또한 support barrier는 RLVR가 “없던 능력을 창조하는 기계”가 아니라 “가끔 나오는 정답 경로를 증폭하는 기계”에 가깝다는 점을 명확히 한다. 이 해석은 후속 실험과 practical takeaway를 이해하는 데 필수적이다. (근거: Section 3)

---
# 4. LLM as a Multi-Armed Bandit 확장 해설

## A. 섹션 길잡이

이 절은 RLVR/GRPO를 sequence-level bandit으로 다시 쓰는 이론적 기초를 제공한다. 핵심 아이디어는 간단하지만 중요하다. completion 전체가 끝난 뒤에 verifier가 점수를 주는 구조라면, 토큰 단위 MDP보다 **완성된 답변 하나를 arm으로 보는 multi-armed bandit 추상화**가 더 직접적이라는 것이다. 그러나 실제 completion space는 방대하므로, 저자들은 이를 반복적으로 등장하는 reasoning mode로 coarse-grain한다. (근거: Section 4, Appendix A)

이 절을 읽을 때는 세 가지를 확인해야 한다. 첫째, 왜 completion 전체가 action unit이 되는가. 둘째, 무한해 보이는 출력 공간이 실제론 왜 유한 support처럼 다뤄질 수 있는가. 셋째, good/bad mass와 내부 모드 조성 \((y,z)\)을 왜 분리하는가. (근거: Section 4)

## B. 원문 내용 재구성

### 4.1 completion-level action과 truncation된 출력 공간

[원문 근거] 고정된 프롬프트 \(x\)에 대해 LLM은 completion \(y\sim \pi_\omega(\cdot\mid x)\)를 샘플한다. RLVR에서 reward는 completion 전체에 대해 한 번 부여되므로, 자연스러운 bandit arm은 “다음 토큰”이 아니라 “완성된 답변 하나”다. (근거: Section 4)

[원문 근거] 이론적으로는 문장 길이에 제한이 없으면 completion space가 무한할 수 있지만, 실제 시스템은 최대 길이 \(L_{\max}\)로 truncation된다. 따라서 허용 가능한 출력 공간은

$$
Y_{\le L_{\max}} = \bigcup_{\ell=1}^{L_{\max}} V^\ell
$$

로 유한하게 잘린다. 논문은 truncation된 정책을

$$
\pi_\omega^{(L)}(y\mid x) \propto \pi_\omega(y\mid x)\mathbf{1}\{y\in Y_{\le L_{\max}}\}
$$

와 같은 형태로 적는다. 즉 실제 training dynamics는 유한 support 위 bandit으로 해석될 수 있다. (근거: Section 4, Appendix A.2)

### 4.2 coarse-graining map과 reasoning mode

[원문 근거] completion 문자열 하나하나를 그대로 arm으로 두면 상태공간이 지나치게 커진다. 이를 줄이기 위해 저자들은 coarse-graining map

$$
\phi: Y_{\le L_{\max}} \to H = \{h_1,\dots,h_{K+M}\}
$$

를 도입한다. 여기서 \(h_i\)는 개별 문자열이 아니라, 서로 다른 표면형을 가지더라도 동일하거나 유사한 해법 패턴을 공유하는 **reasoning mode**를 의미한다. (근거: Section 4, Appendix A.3)

[원문 근거] 이 map에 의해 mode-level policy가 유도된다.

$$
\pi_\theta(h\mid x)=\sum_{y:\phi(y)=h} \pi_\omega^{(L)}(y\mid x),
\qquad
\pi_\theta(h_i\mid x)=\mathrm{softmax}(\theta)_i.
$$

여기서 \(\theta\)는 원래 LLM의 모든 파라미터를 대신하는 effective logit 좌표로 이해된다. 저자들은 이후 분석을 이 저차원 mode distribution 위에서 전개한다. (근거: Section 4)

### 4.3 good mode와 bad mode의 분해

[원문 근거] 저자들은 mode 집합을 good set과 bad set으로 나눈다.

$$
H = H^+ \cup H^-,
\qquad |H^+|=K,\quad |H^-|=M.
$$

여기서 good mass와 bad mass는 각각

$$
\alpha = \sum_{h\in H^+}\pi_\theta(h\mid x),
\qquad
p = \sum_{h\in H^-}\pi_\theta(h\mid x)=1-\alpha
$$

로 정의된다. 이후 논문 전반에서 핵심 상태변수로 쓰이는 것은 바로 \(p\), 즉 bad mode 총질량이다. (근거: Section 4)

### 4.4 good 내부 조성과 bad 내부 조성

[원문 근거] 전체 good mass와 bad mass 외에도, 각각의 내부 조성을 따로 정규화하여

$$
y_i = \frac{\pi_\theta(h_i\mid x)}{\alpha} \quad (h_i\in H^+),
\qquad
z_j = \frac{\pi_\theta(h_j\mid x)}{p} \quad (h_j\in H^-)
$$

를 정의한다. 따라서 전체 분포는

$$
(\alpha y_1,\dots,\alpha y_K,\; p z_1,\dots,p z_M)
$$

형태로 쓸 수 있다. 이 분해의 의미는 중요하다. 한 층에서는 good과 bad 사이 총질량이 어떻게 이동하는지를 보아야 하고, 다른 층에서는 good들끼리 혹은 bad들끼리 누가 지배적인 mode가 되는지를 보아야 하기 때문이다. (근거: Section 4)

### 4.5 극한적 해석과 coarse-graining의 위상적 역할

[원문 근거] 논문은 \(\phi\)가 항등이면 개별 문자열 하나하나가 arm인 극단을, \(L_{\max}\to\infty\)면 이론상 무한 출력 공간을 가정하는 극단을 생각할 수 있다고 설명한다. 그러나 reasonable한 coarse-graining을 택하는 한, 이후에 도출되는 핵심 결과—bad mass를 지배하는 \(J\)의 부호, good 내부 winner-take-all, bad 내부 spreading—는 크게 변하지 않는다고 본다. (근거: Section 4)

## C. 친절한 직관과 해설

[추가 설명(일반 지식)] 이 절은 실무적으로도 중요하다. 실제로 LLM이 생성하는 completion은 표면형이 무수히 많지만, 많은 경우 이들은 소수의 반복되는 해법 패턴으로 묶인다. 예를 들어 코딩 문제라면 “정확한 알고리즘 사용”, “입력 파싱은 맞지만 핵심 로직 오류”, “부분 문제만 해결”, “테스트를 우연히 통과” 같은 모드가 존재할 수 있다. 논문은 이런 모드들의 확률질량이 어떻게 진화하는지를 추적한다. 따라서 이후의 모든 식은 토큰 선택보다는 **해법 군집의 경쟁과 재배치**로 읽어야 이해가 쉽다. (근거: Section 4)

[원문 근거] 한 가지 유의할 점은, 논문이 reasoning mode \(\phi\)를 실험적으로 실제 클러스터링했다는 뜻은 아니라는 점이다. mode 분해는 주로 분석을 위한 추상화로 제시되며, 구체적인 자동 clustering 절차는 본문과 부록에 명시되어 있지 않다. (근거: Section 4, Appendix A.3)

---
# 5. Geometry of the Probability Simplex 확장 해설

## A. 섹션 길잡이

Section 5는 본 논문의 수학적 핵심부다. 저자들은 mode distribution이 놓인 공간이 단순한 Euclidean 공간이 아니라, 합이 1로 고정된 **확률단체(simplex)** 라는 사실을 강조한다. softmax Jacobian은 이 공간의 자연 기하를 만들고, GRPO mean-field dynamics는 이 기하 위에서 replicator flow 또는 natural gradient flow로 해석된다. (근거: Section 5, Figure 4, Appendix H)

이 절에서 반드시 짚어야 할 것은 다음과 같다. 첫째, softmax Jacobian \(J(p)=\mathrm{Diag}(p)-pp^\top\)의 의미. 둘째, good block과 bad block으로 나누었을 때 dynamics가 \(y,z,p\) 세 부분으로 분해된다는 사실. 셋째, good block 내부에서는 왜 diversity collapse가, bad block 내부에서는 왜 entropy increase가 발생하는가. 넷째, finite rollout이 왜 Wright–Fisher형 drift noise처럼 해석되는가. (근거: Section 5)

## B. 원문 내용 재구성

### 5.1 확률단체와 Figure 4의 요지

> [Figure 삽입 안내] 이 위치에 **논문 원문의 Figure 4** crop을 넣으십시오.  
> - 권장 캡션: `5.1 확률단체와 Figure 4의 요지`  
> - 편집 원칙: 도판 자체는 원문 이미지를 사용하고, 본문에서는 **이 도판이 무엇을 보여 주는지**만 해설합니다.

[원문 근거] 상태벡터는

$$
p=(p_1,\dots,p_K,p_{b_1},\dots,p_{b_M})\in \Delta^{K+M-1}
$$

에 놓인다. 여기서

$$
\Delta^d = \{x\in\mathbb{R}^{d+1}_{\ge 0}: \mathbf{1}^\top x = 1\}
$$

는 확률단체이고, 총질량 보존 때문에 허용 가능한 속도벡터는

$$
T_p\Delta^{K+M-1}=\{v\in\mathbb{R}^{K+M}:\mathbf{1}^\top v=0\}
$$

라는 접공간에 존재해야 한다. Figure 4는 바로 이 비유클리드 공간 위에서 정책이 움직인다는 점을 도식화한다. (근거: Section 5, Figure 4)

[원문 근거] Figure 4 캡션은 softmax Jacobian

$$
J(p)=\mathrm{Diag}(p)-pp^\top
$$

이 Shahshahani/Fisher geometry를 부여한다고 설명한다. 또한 forward KL의 국소 이차형은

$$
\frac12\delta^\top \mathrm{Diag}(p)^{-1}\delta
$$

로 주어져, 확률질량 이동의 비용이 좌표별로 비균등함을 드러낸다. 희귀한 arm에서의 작은 질량 이동과 자주 선택되는 arm에서의 동일한 질량 이동은 같은 Euclidean 이동이 아니며, 이것이 자연그래디언트 해석의 배경이 된다. (근거: Figure 4, Appendix H)

### 5.1.1 softmax Jacobian의 기본 작용

[원문 근거] 저자들은 임의의 벡터 \(v\)에 대해 softmax Jacobian의 작용을

$$
J(p)v = p\odot (v-\bar v\mathbf{1}),
\qquad \bar v = p^\top v
\tag{10}
$$

로 쓴다. 여기서 \(\odot\)는 elementwise product다. 이 식은 두 가지 중요한 구조를 노출한다. 첫째, 각 좌표의 변화량에는 현재 질량 \(p_i\)가 곱해진다. 따라서 \(p_i=0\)이면 해당 좌표는 움직일 수 없다. 둘째, 평균 \(\bar v\)를 빼므로 총합이 자동으로 보존된다. (근거: Section 5, Eq. 10)

### 5.1.2 replicator-form의 GRPO dynamics

[원문 근거] 연속시간에서 GRPO mean-field flow는

$$
\dot p = \eta J(p)^2 A
$$

로 쓸 수 있고, 동치로

$$
\dot p
=
\eta\,p\odot\Big[J(p)A - \langle p, J(p)A\rangle\mathbf{1}\Big]
\tag{11}
$$

의 replicator-form으로 표현된다. 저자들은 이 식에서 두 가지를 강조한다. 하나는 **multiplicativity**다. 각 변화량 앞에 \(p_i\)가 곱해지므로 support face가 보존된다. 다른 하나는 **relative performance**다. 질량 이동은 절대 점수가 아니라 평균 대비 우위에 의해 결정된다. (근거: Section 5, Eq. 11)

### 5.1.3 good/bad block 분해: 식 (12a)–(12c)

[원문 근거] 전체 분포를 \(((1-p)y,\; pz)\)로 분해하면, dynamics는 세 개의 식으로 정리된다.

$$
\dot y = +\kappa(p)\, y\odot\big(y-\|y\|_2^2\mathbf{1}\big),
\tag{12a}
$$

$$
\dot z = -\kappa(p)\, z\odot\big(z-\|z\|_2^2\mathbf{1}\big),
\tag{12b}
$$

$$
\dot p = -\eta\frac{J}{\sigma(p)}[p(1-p)]^2\big(\|y\|_2^2+\|z\|_2^2\big),
\tag{12c}
$$

여기서

$$
\kappa(p)=\eta\frac{J}{\sigma(p)}p(1-p).
$$

즉 \(y\)와 \(z\)는 내부 조성 dynamics이고, \(p\)는 전체 bad mass dynamics다. (근거: Section 5.1, Eq. 12a–12c)

[원문 근거] noise-free case에서는 \(J=1\), \(\sigma(p)=\sqrt{p(1-p)}\)가 되어

$$
\kappa(p)=\eta\sqrt{p(1-p)}
$$

로 단순화된다. 저자들의 해석은 명확하다. **노이즈는 simplex 기하 자체를 바꾸지 않는다. 대신 시간척도와 방향의 부호를 바꾼다.** (근거: Remark 5.1)

### 5.1.4 good block 내부: winner-take-all과 diversity collapse

[원문 근거] \(J>0\)이면 \(\kappa(p)>0\)이므로 good block dynamics는

$$
\dot y = \kappa(p)\, y\odot\big(y-\|y\|_2^2\mathbf{1}\big)
$$

를 따른다. 따라서 \(y_i>\|y\|_2^2\)인 좌표는 증가하고, \(y_i<\|y\|_2^2\)인 좌표는 감소한다. 초기 good arm들 사이에 아주 미세한 비대칭만 있어도, 상대적으로 큰 좌표가 계속 강화되어 결국 한 arm이 지배적이 된다. 저자들은 이를 Figure 2와 연결해 **winner-take-all** 구조라고 설명한다. (근거: Section 5.1, Figure 2)

[추가 설명(일반 지식)] 여기서 주의할 점은 collapse의 대상이 “정답/오답 전체”가 아니라 **good mode 내부의 다양성**이라는 점이다. 즉 최종 정확도는 상승할 수 있지만, 동일하게 정답인 여러 reasoning path가 동시에 보존되지는 않는다. 이것이 논문이 diversity collapse를 구조적 귀결로 강조하는 이유다. (근거: Section 5.1, Appendix I, Appendix K)

### 5.1.5 bad block 내부: entropy increase와 균등화

[원문 근거] bad block dynamics는 부호가 반대이므로

$$
\dot z = -\kappa(p)\, z\odot\big(z-\|z\|_2^2\mathbf{1}\big)
$$

가 된다. 따라서 \(J>0\)에서는 bad mode의 내부 concentration이 약해지고, \(z\)는 점점 균등분포 \(\frac1M\mathbf{1}\) 쪽으로 퍼진다. 논문은 이를 maximum-entropy 방향의 spreading으로 해석한다. 나쁜 모드 질량 자체는 줄어들지만, 남아 있는 bad mass는 소수의 bad arm에 몰리기보다 여러 bad arm에 diffuse되는 경향을 보인다는 것이다. (근거: Section 5.1, Appendix J)

### 5.1.6 bad mass \(p\)의 드리프트와 구조적 sparsity 인자

[원문 근거] 식 (12c)는 bad mass 감소율이 단지 \(J\)와 \([p(1-p)]^2\)뿐 아니라,

$$
\|y\|_2^2 + \|z\|_2^2
$$

라는 구조적 인자에 의해 조절된다는 점을 보여준다. 이는 good과 bad 내부가 얼마나 concentrated되어 있는지를 반영한다. late time에서 good block은 보통 vertex로 collapse하므로 \(\|y\|_2^2\to 1\)이 되고, bad block은 uniform으로 spread되므로 \(\|z\|_2^2\to 1/M\)이 된다. 따라서 long-time regime에서는 이 기하 인자가 비교적 안정된 값으로 정리된다. (근거: Section 5.1, Appendix I, Appendix J)

### 5.2 Shahshahani metric과 자연그래디언트

[원문 근거] Section 5.2에서 저자들은 simplex의 자연 기하를 Shahshahani metric으로 정의한다.

$$
\langle u,v\rangle_{\mathrm{Shah};p}
=
\sum_i \frac{u_i v_i}{p_i},
\qquad u,v\in T_p\Delta.
$$

이는 categorical family의 Fisher metric과 같은 구조를 갖는다. 이 metric 하에서 함수 \(F(p)\)의 natural gradient는

$$
\mathrm{grad}_{\mathrm{Shah}}F(p)
=
J(p)\nabla F(p)
=
p\odot\big(\nabla F(p)-\langle p,\nabla F(p)\rangle\mathbf{1}\big)
$$

로 주어진다. (근거: Section 5.2, Appendix H)

[원문 근거] good block에서 potential을

$$
\Phi(y):=\frac12\|y\|_2^2
$$

로 두면,

$$
\dot y = \kappa(p)\,\mathrm{grad}_{\mathrm{Shah}}\Phi(y)
$$

가 된다. 즉 good block dynamics는 Herfindahl index, 곧 concentration index를 증가시키는 방향의 natural gradient ascent다. 이것이 good mode diversity collapse를 geometric language로 재해석한 결과다. (근거: Section 5.2, Appendix K)

### 5.2.1 KL-regularized mirror ascent와의 연결

[원문 근거] 저자들은 discrete-time viewpoint에서도 동일한 구조를 보인다. 다음의 KL-regularized 선형화 문제를 생각하자.

$$
p^+
=
\arg\max_{q\in\Delta}
\left\{
\langle A,q\rangle - \frac1\eta D_{KL}(q\|p)
\right\}.
\tag{13}
$$

그 해는

$$
p_i^+ = \frac{p_i e^{\eta A_i}}{\sum_j p_j e^{\eta A_j}}
$$

와 같은 multiplicative-weights update가 되며, 1차 근사하면

$$
p^+-p = \eta\,p\odot(A-\langle p,A\rangle\mathbf{1}) + O(\eta^2)
$$

을 얻는다. 이는 앞선 replicator step과 일치한다. 저자들이 강조하는 요점은 다음과 같다. **mirror ascent, entropic update, natural gradient flow는 모두 확률단체의 동일한 기하적 구조에서 만난다.** (근거: Section 5.2, Eq. 13, Appendix H)

### 5.3 finite sampling과 genetic drift

[원문 근거] 지금까지의 분석은 mean-field drift를 가정했지만, 실제 GRPO는 프롬프트마다 \(G\)개의 rollout만 샘플링한다. 따라서 empirical mean과 variance로 advantage를 계산하는 순간 표본잡음이 추가된다. 저자들은 이를 진화동역학 비유를 따라 **genetic drift**라고 부른다. (근거: Section 5.3)

[원문 근거] \(G\)개의 categorical draw에서 empirical frequency \(\hat y\)는

$$
\sqrt{G}(\hat y-y) \Rightarrow \mathcal{N}(0,\mathrm{Diag}(y)-yy^\top)
$$

를 따른다. simplex 전체의 one-hot score feature 평균 covariance는

$$
\frac{1}{G}\Sigma(p),
\qquad
\Sigma(p)=\mathrm{Diag}(p)-pp^\top
$$

가 된다. 따라서 diffusion approximation은

$$
dp = \dot p\,dt + \eta\frac{\sqrt{\nu}}{\sqrt{G}}\Sigma(p)^{1/2}dW_t
$$

형태의 Wright–Fisher형 noise로 쓸 수 있다. 저자들은 noisy reward나 advantage normalization이 주로 diffusion amplitude를 재조정할 뿐, simplex-shaped covariance 구조 자체는 그대로 유지된다고 설명한다. (근거: Section 5.3)

## C. 친절한 직관과 해설

[추가 설명(일반 지식)] Section 5를 한 문장으로 요약하면, **GRPO는 확률단체 위 자연선택형 질량 재배치**다. 확률벡터는 합이 1이어야 하므로 일반 Euclidean gradient descent처럼 움직일 수 없고, softmax Jacobian이 그 제약을 만족시키는 자연스러운 projector 역할을 한다. 그 결과 “평균보다 더 나은 모드가 늘어나고, 평균보다 못한 모드는 줄어드는” replicator 구조가 발생한다. 또한 good mode들이 여러 개 있을 때도 균형 공존이 아니라 하나의 dominant mode로 편중되기 쉽다는 점은, RLVR를 reasoning quality 향상 도구로만 볼 것이 아니라 **output diversity를 잠식할 수 있는 알고리즘**으로도 보아야 함을 시사한다. (근거: Section 5, Appendix I–K)

---
# 6. LLM as a Bandit under Noisy Rewards 확장 해설

## A. 섹션 길잡이

이 절은 이론을 실제 PPO/GRPO 세부와 연결한다. 실제 시스템에서는 importance sampling ratio, PPO clipping, KL regularization 같은 요소들이 존재한다. 따라서 앞서 도출한 sign-of-\(J\) phase transition이 이러한 장치들 아래에서도 유지되는지 확인하는 것이 필요하다. 저자들의 결론은 분명하다. 작은 스텝 크기에서는 clipping과 importance sampling은 leading-order drift를 바꾸지 않으며, KL regularization은 sharp boundary collapse를 interior equilibrium으로 완화하지만 \(J\)의 부호 자체를 무력화하지는 못한다. (근거: Section 6, Appendix F, Appendix G)

## B. 원문 내용 재구성

### 6.1 clipping과 importance sampling의 영향

[원문 근거] 저자들은 실제 GRPO가 PPO-style importance ratio와 clipping을 사용한다는 점을 인정한다. 그러나 small-step regime, 즉 학습률 \(\eta\)가 clip threshold \(\epsilon,\epsilon'\)에 비해 충분히 작다면, clipping과 importance sampling에서 생기는 보정항은 \(O(\eta^2)\) remainder에 머무른다고 주장한다. 다시 말해, mean-field ODE의 leading-order drift는 앞서 도출한 구조를 그대로 유지한다. (근거: Section 6 도입, Appendix F)

[원문 근거] Appendix F의 논리 전개는 다음과 같다. exact importance ratio를

$$
\rho_i = \frac{\pi_{\mathrm{new}}(i)}{\pi_{\mathrm{old}}(i)} = \frac{p_i^+}{p_i}
$$

로 놓으면, small-step에서 \(\|p^+-p\|=O(\eta)\)이므로 \(\rho_i=1+O(\eta)\)다. 이를 update 식에 대입하면 추가 correction term이 생기지만, softmax pushforward까지 고려한 후에는 모두 \(O(\eta^2)\)에 포함된다. 따라서 phase portrait의 1차 구조는 변하지 않는다. (근거: Appendix F)

### 6.2 Theorem 6.1: multi-bad-arm에서의 실전형 bad-mass dynamics

[원문 근거] 여러 bad arm이 존재하는 일반 설정에서 저자들은

$$
p(t)=\sum_{m=1}^{M} p_{b_m}(t),
\qquad
y_j(t)=\frac{p_j(t)}{1-p(t)},
\qquad
z_m(t)=\frac{p_{b_m}(t)}{p(t)}
$$

를 두고, collision masses를

$$
s_2(t)=\|y(t)\|_2^2\in[1/K,1],
\qquad
t_2(t)=\|z(t)\|_2^2\in[1/M,1]
$$

로 정의한다. 이어서 geometry factor를

$$
C_{\mathrm{geo}}(t):=s_2(t)+t_2(t)\in[1/K+1/M,2]
$$

로 둔다. (근거: Theorem 6.1 직전 정의)

[원문 근거] 그러면 Theorem 6.1은 다음을 준다.

$$
\dot p(t)
=
-\eta\,\frac{J}{\sigma(p(t))}[p(t)(1-p(t))]^2\,C_{\mathrm{geo}}(t)
+ O(\eta^2).
\tag{14}
$$

즉 Section 3의 단일 bad-mass 식이 multiple good/bad arm 구조까지 반영한 일반형으로 확장된다. 중요하게도 방향은 여전히 \(\mathrm{sign}(J)\)가 결정하고, 내부 조성은 \(C_{\mathrm{geo}}(t)\)라는 양의 기하 인자를 통해 속도에만 관여한다. (근거: Theorem 6.1, Eq. 14)

### 6.2.1 internal time과 logit form

[원문 근거] 저자들은 bad mass logit을

$$
L(t):=\log\frac{p(t)}{1-p(t)}
$$

로 두고, 내부 시간(internal time)을

$$
\tau(t)=\int_0^t \eta\frac{|J|}{\sigma(p(u))}p(u)(1-p(u))\,du
\tag{15}
$$

로 정의한다. 그러면 logit dynamics는

$$
\frac{dL}{d\tau}
=
-\mathrm{sign}(J)\,C_{\mathrm{geo}}(\tau)
=
-\mathrm{sign}(J)\big(s_2(\tau)+t_2(\tau)\big)
\tag{16}
$$

이 된다. 여기서 \(C_{\mathrm{geo}}(\tau)>0\)이므로, \(L\)의 증가·감소는 오직 \(\mathrm{sign}(J)\)에 의해 결정된다. 따라서 practical GRPO에서도 learning, neutral, anti-learning의 phase structure는 유지된다. (근거: Section 6, Eq. 15–16)

### 6.2.2 small-heterogeneity regime

[원문 근거] 초기 \(y\)와 \(z\)가 거의 균등한 블록 조성이라고 가정하면,

$$
y(0)=u_K+v_0,
\qquad
z(0)=u_M+w_0
$$

와 같이 쓸 수 있다. 여기서 이질성의 크기는

$$
\zeta_0=\|v_0\|_2^2=s_2(0)-1/K,
\qquad
\xi_0=\|w_0\|_2^2=t_2(0)-1/M
$$

로 정의된다. 그러면 bad-mass logit은

$$
L(\tau)
=
L(0)
-\mathrm{sign}(J)\Big(\frac1K+\frac1M\Big)\tau
-\mathrm{sign}(J)\frac K2 \zeta_0(e^{2\tau/K}-1)
-\mathrm{sign}(J)\frac M2 \xi_0(1-e^{-2\tau/M})
+ R_L(\tau)
\tag{17}
$$

로 전개된다. 저자들이 강조하는 바는, 작은 이질성 영역에서는 초기 세부 분포 전체보다 \((\zeta_0,\xi_0)\)라는 두 개의 스칼라가 1차적으로 중요하다는 점이다. (근거: Section 6, Eq. 17, Theorem I.7)

### 6.3 KL regularization

[원문 근거] 저자들은 two-class bad mass 관점에서 KL penalty를 다음과 같이 도입한다.

$$
\dot p_{\mathrm{KL}}
=
-\beta p(1-p)
\left(
\log\frac{p}{1-p}-\log\frac{p_{\mathrm{ref}}}{1-p_{\mathrm{ref}}}
\right).
$$

reward-driven drift와 합치면

$$
\dot p
=
-\eta\frac{J}{\sigma(p)}[p(1-p)]^2 C(y,z)
-
\beta p(1-p)
\left(
\log\frac{p}{1-p}-\log\frac{p_{\mathrm{ref}}}{1-p_{\mathrm{ref}}}
\right)
\tag{18}
$$

이 된다. full reverse-KL을 사용할 경우에는 여기에 \(D_{KL}(y\|y^{\mathrm{ref}})\)와 \(D_{KL}(z\|z^{\mathrm{ref}})\) 차이가 추가적으로 들어간다. (근거: Section 6.1, Eq. 18, Appendix G)

### 6.3.1 interior fixed point의 존재와 성질

[원문 근거] interior equilibrium \(p^\star\in(0,1)\)는

$$
\beta\Big(L(p^\star)-L(p_{\mathrm{ref}})\Big)
=
-\eta\frac{J}{\sigma(p^\star)}p^\star(1-p^\star)C(y,z),
\qquad
L(p)=\log\frac{p}{1-p}
\tag{19}
$$

을 만족한다. 저자들은 \(\beta>0\)이고 \((y,z)\)가 고정되면 이 fixed point가 유일하고 전역적으로 안정하다고 정리한다. 그 위치는 여전히 \(J\)의 부호를 따른다.

- \(J>0\): \(0<p^\star<p_{\mathrm{ref}}\)
- \(J=0\): \(p^\star=p_{\mathrm{ref}}\)
- \(J<0\): \(p_{\mathrm{ref}}<p^\star<1\)

즉 KL regularization은 reward-driven collapse를 없애는 것이 아니라, 경계 극점 대신 interior equilibrium으로 **완화**한다. (근거: Section 6.1, Eq. 19, Theorem G.6)

### 6.3.2 강한 KL과 약한 KL의 극한

[원문 근거] 강한 KL 영역에서는 equilibrium이 reference policy 근방으로 붙는다. 저자들은

$$
p^\star
\approx
p_{\mathrm{ref}} - \frac{\eta J}{\beta}
\frac{[p_{\mathrm{ref}}(1-p_{\mathrm{ref}})]^2}{\sigma(p_{\mathrm{ref}})}C(y,z)
$$

와 같은 \(O(\beta^{-1})\) 근사를 제시한다. 반대로 약한 KL에서 \(J<0\)인 경우,

$$
1-p^\star \sim \frac{\beta}{c}\log\frac{c}{\beta},
\qquad c=-\eta\frac{J}{\sigma(1)}C(y,z)>0
$$

이 되어, 아주 작은 양의 \(\beta\)라도 total collapse \((p=1)\)는 막을 수 있음을 보여준다. (근거: Section 6.1, Appendix G)

### 6.3.3 Figure 5의 의미

> [Figure 삽입 안내] 이 위치에 **논문 원문의 Figure 5** crop을 넣으십시오.  
> - 권장 캡션: `6.3.3 Figure 5의 의미`  
> - 편집 원칙: 도판 자체는 원문 이미지를 사용하고, 본문에서는 **이 도판이 무엇을 보여 주는지**만 해설합니다.

[원문 근거] Figure 5는 KL regularization이 sharp phase transition을 smoothing하지만, \(J\) 의존성을 제거하지는 않는다는 점을 시각화한다. \(J>0\)이면 equilibrium은 reference보다 낮은 bad mass 쪽에, \(J<0\)이면 더 높은 bad mass 쪽에 위치한다. \(\beta\to\infty\)이면 reference policy로 수축하고, \(\beta\to 0\)이면 reward-driven boundary dynamics에 가까워진다. (근거: Figure 5, Section 6.1)

## C. 친절한 직관과 해설

[추가 설명(일반 지식)] Section 6의 직관은 “reward signal”과 “anchoring force”의 줄다리기다. reward-driven term은 verifier가 가리키는 방향으로 질량을 이동시키고, KL term은 reference policy에서 너무 멀어지지 않도록 잡아당긴다. 그러나 verifier가 근본적으로 anti-informative \((J<0)\)라면, KL은 학습 방향을 반전시키지 못한다. 다만 그 잘못된 방향으로의 붕괴를 완화해 interior equilibrium을 만든다. 이는 실무에서 KL regularization을 안정화 도구로 사용할 수는 있지만, verifier 품질 문제를 대체할 수는 없다는 결론으로 이어진다. (근거: Section 6, Appendix G)

---
# 7. Experiments 확장 해설

## A. 섹션 길잡이

실험의 목적은 이론의 두 핵심 가설을 검증하는 데 있다. 첫째, \(J=0\) 부근에서 실제 RL fine-tuning에서도 sharp phase transition이 관찰되는가. 둘째, \(J>0\) 영역에서 noise는 fate가 아니라 rate만 바꾸는가. 즉 verifier가 informative하기만 하면, 노이즈가 존재하더라도 학습은 같은 방향으로 진행되는가. (근거: Section 7.1)

이 절은 이론 파트와 달리 구체적이다. 데이터셋, base model, training steps, rollout 수, KL coefficient, noise injection 방식, 평가 metric, 표와 그림의 정량 결과를 모두 확인해야 한다. (근거: Section 7)

## B. 원문 내용 재구성

### 7.1 Experimental Hypotheses

[원문 근거] 저자들은 두 개의 가설을 명시한다.

- **H1**: \(J=\mathrm{TPR}-\mathrm{FPR}>0\)일 때만 accuracy가 개선된다. \(J=0\)은 neutral drift, \(J<0\)은 anti-learning이다.
- **H2**: \(J>0\)이면 noisy reward와 clean reward는 동일한 basin of attraction을 갖고, noise는 convergence rate만 바꾼다.

즉 실험은 단순한 성능 비교가 아니라, 이론이 제안한 phase transition과 rate-versus-fate 분리를 검증하기 위한 설계다. (근거: Section 7.1)

### 7.2 Setup

[원문 근거] 실험 과제는 **programmatic verification이 가능한 Python code generation**이다. 데이터는 OpenR1 Hugging Face의 high-quality subset에서 구성되며,

$$
N_{\mathrm{train}}=10{,}239,
\qquad
N_{\mathrm{val}}=594
$$

이다. 각 샘플에는 자연어 명세, 입력/출력 예시, public 및 hidden unit test harness가 포함된다. (근거: Section 7.2)

[원문 근거] base model은 **Qwen2.5-3B**이고, 핵심 평가 지표는 validation set에서의 \(\mathbb{E}[\mathrm{pass@1}]\)이다. 각 하이퍼파라미터 설정마다 독립적으로 5회 실행한 결과를 평균하여 보고한다. (근거: Section 7.2)

[원문 근거] training은 VeRL 기반 standard GRPO를 사용하며, per-group advantage standardization과 importance-ratio clipping이 포함된다. 각 prompt마다 \(G=8\)개의 rollout을 생성하고, return은 advantage 계산 전에 zero mean과 unit variance로 정규화된다. KL penalty coefficient는 **\(\beta=0\)** 으로 고정되어, reward-driven dynamics를 직접 관찰하도록 설계되었다. (근거: Section 7.2, Appendix L)

[원문 근거] synthetic noise injection은 oracle correctness \(z\in\{0,1\}\)에 noisy checker reward \(r\in\{0,1\}\)를 얹는 방식으로 구현된다. 즉,

- \(z=1\)이면 \(r\sim \mathrm{Bernoulli}(\mathrm{TPR})\)
- \(z=0\)이면 \(r\sim \mathrm{Bernoulli}(\mathrm{FPR})\)

로 샘플링한다. 따라서

$$
J=\mathrm{TPR}-\mathrm{FPR}
$$

이며, 논문은 \(J\in[-0.1,1]\) 범위를 탐색한다. 또한 동일한 \(J\) 값에 대해 여러 \((\mathrm{TPR},\mathrm{FPR})\) 조합을 비교하여, noise magnitude와 noise structure의 효과를 분리한다. (근거: Section 7.2, Appendix M)

[원문 근거] training protocol은 총 **2 epochs = 1,410 gradient steps**이며, 5 step마다 metric을 로깅한다. 각 설정에 대해 5개의 random seed를 사용하고 평균과 표준편차를 보고한다. 다른 하이퍼파라미터는 noise condition 간 고정된다. baseline은 noise-free oracle, 즉 \(J=1\)이다. GPU 종류, 전체 wall-clock time 등은 본문과 Appendix L에 명시되어 있지 않다. (근거: Section 7.2, Appendix L)

### 7.3 Results

[원문 근거] Table 1은 각 \(J\) 및 \((\mathrm{FPR},\mathrm{FNR})\) 조합에서의 최종 validation 성능을 정리한다. 값은 다음과 같다.

| \(J\) | \((\mathrm{FPR},\mathrm{FNR})\) | \(\mathbb{E}[\mathrm{pass@1}]\) | Improvement |
|---|---:|---:|---:|
| -0.1 | (0.60, 0.50) | 0.16% | -12.6% |
| 0.0 | (0.50, 0.50) | 13.40% | +0.6% |
| 0.3 | (0.00, 0.70) | 16.00% | +3.2% |
| 0.3 | (0.70, 0.00) | 14.60% | +1.8% |
| 0.7 | (0.20, 0.10) | 18.6% | +5.8% |
| 1.0 | (0.00, 0.00) | 20.8% | +8.0% |

(근거: Table 1)

[원문 근거] 저자들의 해석은 다음과 같다. \(J>0\)인 모든 조건에서 validation accuracy는 훈련 중 상승한다. \(J\)가 클수록 향상 폭과 속도가 모두 커진다. \(J=0\)에서는 개선폭이 +0.6%에 그쳐 사실상 neutral drift와 정합적이다. \(J<0\)에서는 성능이 크게 하락하며, 이는 anti-learning과 collapse를 뒷받침한다. Figure 1의 학습 곡선은 이 정량 결과를 시간축 위에 시각화한 것이다. (근거: Section 7.3, Figure 1, Table 1)

[원문 근거] 저자들은 H2, 즉 “rate, not fate” 가설에 대해서도 논의한다. 실험 horizon이 1,410 step으로 유한하기 때문에 완전한 점근영역을 직접 확인한 것은 아니지만, 모든 \(J>0\) 곡선이 training 동안 일관되게 상승한다는 점은 noisy reward가 basin을 바꾸기보다는 수렴 속도를 늦춘다는 해석과 부합한다고 본다. 특히 \(J=0.3\)처럼 signal quality가 크게 degraded된 경우에도 여전히 학습은 진행되지만, 훨씬 느리다. (근거: Section 7.3)

[원문 근거] 동일한 \(J=0.3\)에서도 noise structure 차이가 관찰된다. 본문 서술은 \((\mathrm{FPR}=0.00, \mathrm{FNR}=0.70)\)에서 약 15.98%, \((\mathrm{FPR}=0.70, \mathrm{FNR}=0.00)\)에서 약 14.64%라고 적고, Table 1은 이를 각각 16.00%와 14.60%로 반올림해 제시한다. 방향성 자체는 일관되다. **동일한 \(J\)** 여도 FPR이 큰 경우가 더 해롭다. 저자들은 이를 이론의 비대칭적인 rate law와 연결해 해석한다. (근거: Section 7.3 본문, Table 1)

### 7.4 Limitations and Future Directions

[원문 근거] 첫 번째 한계는 **oracle imperfection**이다. unit test suite가 finite하기 때문에 \((\mathrm{TPR},\mathrm{FPR})\)의 실제 추정값 자체가 systematic bias를 가질 수 있다. hidden edge case가 많은 코딩 과제에서는 이 문제가 특히 중요하다. (근거: Section 7.4)

[원문 근거] 두 번째 한계는 **context length effect**다. 특히 \(J<0\)에서 모델이 성능 저하와 함께 더 긴 응답을 생성하고 max token length를 초과할 수 있는데, VeRL truncation은 이런 경우 reward 0과 clipping ratio 증대를 유발해 사실상 false negative를 추가로 늘릴 수 있다. 저자들은 이것이 anti-learning 영역에서 이론과 실험의 일부 비대칭을 설명할 수 있다고 본다. (근거: Section 7.4)

[원문 근거] 세 번째 한계는 **generalization**이다. 실험은 Python coding과 Qwen2.5-3B에 집중되어 있다. 저자들은 \(J=0\) 경계 자체는 learning dynamics의 보다 근본적인 성질이므로 다른 과제와 모델에도 나타날 가능성이 높다고 보지만, decay rate나 noise tolerance의 수치적 상세는 task complexity, model size, verifier 특성에 따라 달라질 수 있다고 적는다. 수학 reasoning, creative writing + LLM judge, 더 큰 모델이 향후 과제로 제시된다. (근거: Section 7.4)

[원문 근거] 네 번째 한계는 **time-dependent noise**다. 이론은 \(\mathrm{TPR}(t),\mathrm{FPR}(t)\)처럼 시간에 따라 변하는 verifier 품질을 수용할 수 있지만, 실험에서는 고정 synthetic noise만 사용했다. 실제로는 policy와 verifier가 함께 변하며 reward quality도 drift할 수 있으므로, 이러한 공동진화(co-evolution) 분석은 후속 연구 과제로 남겨진다. (근거: Section 7.4)

## C. 친절한 직관과 해설

[추가 설명(일반 지식)] 실험 파트의 강점은 단순한 “노이즈가 많을수록 성능이 덜 오른다”는 수준을 넘어선다는 데 있다. 이 논문은 실제 RL fine-tuning에서도 \(J\)의 부호가 바뀌는 순간 학습 곡선의 방향 자체가 바뀐다는 점을 보여주려 한다. 이는 verifier quality가 training stability의 부차적 요소가 아니라, **학습 가능성과 붕괴를 가르는 구조적 변수**라는 점을 실험적으로 뒷받침한다. 또한 동일한 \(J\)에서도 FPR-heavy와 FNR-heavy 구조가 다른 결과를 낳는다는 관찰은, 단일 스칼라 \(J\)가 phase를 결정하더라도 finite-time regime에서는 noise의 세부 구조가 여전히 중요할 수 있음을 시사한다. (근거: Section 7)

---

# 8. Conclusion 확장 해설

## A. 섹션 길잡이

결론 절은 논문의 핵심 메시지를 다시 단일 문장으로 고정한다. **불완전한 verifier 아래 RLVR의 qualitative outcome은 결국 \(J=\mathrm{TPR}-\mathrm{FPR}\)의 부호에 의해 지배된다.** 이 절에서는 그동안 도출한 이론적 결과와 실험적 관찰을 한 번에 종합하고, practical takeaway를 명시한다. (근거: Section 8)

## B. 원문 내용 재구성

[원문 근거] 저자들은 group-normalized policy-gradient method, 특히 GRPO에서 noisy verifier의 효과를 분석한 결과, 전체 dynamics를 지배하는 핵심 스칼라가 Youden’s index \(J\)라는 점을 결론부에서 재차 강조한다. \(J>0\)이면 verifier가 net-informative하므로 incorrect mode 질량은 감소하고 learning이 일어난다. \(J=0\)이면 chance-level signal만 남아 neutral drift가 된다. \(J<0\)이면 verifier가 anti-informative하여 incorrect mode가 증폭되고 anti-learning과 collapse가 발생한다. (근거: Section 8)

[원문 근거] 결론 절의 “Main Finding” 요약은 다음과 같다. \(J>0\)이면 noisy reward 아래에서도 group-normalized RL은 directionally consistent하며, bad-mode mass는 단조감소하고 accuracy는 증가한다. 이 경우 noise는 주로 progress speed를 늦출 뿐, eventual basin을 바꾸지 않는다. 반대로 \(J<0\)이면 학습 방향이 뒤집혀 오답 모드가 증폭된다. 바로 이것이 “rate, not fate”라는 슬로건의 완성형이다. (근거: Section 8, Main Finding box)

[원문 근거] 저자들이 정리한 논문의 기여는 크게 여섯 가지로 읽을 수 있다.

1. LLM completion-level RLVR를 위한 multi-armed bandit abstraction 제시
2. GRPO dynamics의 mean-field probability-simplex view 정식화
3. bad-mode mass \(p(t)\)라는 지배 상태변수 도출
4. \(t^{-1}\), \(t^{-2}\) tail과 intermediate-difficulty learnability 같은 rate law 제시
5. simplex geometry를 통한 good-manifold 내부 winner-take-all과 diversity collapse 설명
6. practical GRPO detail—PPO clipping, importance sampling, KL regularization—이 drift와 stability에 미치는 영향 분석

(근거: Section 8)

[원문 근거] 결론부는 몇 가지 직접적인 실무 교훈도 적시한다. \(J\)를 먼저 측정해야 하며, \(J\le 0\)라면 compute scaling은 근본 대책이 아니다. \(J>0\)라면 추가 compute는 주로 시간을 산다. false positive는 특히 위험하다. KL regularization은 useful한 stability mechanism이지만, signal quality의 대체재는 아니다. 마지막으로 저자들은 verifier의 판별력 \(J\)가 “fate”를 결정하고, 알고리즘 세부와 noise structure는 주로 “rate”와 stability를 조절한다고 정리한다. (근거: Section 8)

## C. 친절한 직관과 해설

[추가 설명(일반 지식)] 결론의 힘은 복잡한 RLHF/RLVR pipeline의 세부를 하나하나 해체하지 않고도, verifier가 **평균적으로 정답에 더 우호적인가, 오답에 더 우호적인가**라는 질문으로 핵심을 압축했다는 데 있다. 물론 현실 시스템은 time-dependent verifier, prompt별 difficulty 이질성, distribution shift, truncation, mode discovery 문제 등을 포함한다. 그럼에도 이 논문은 최소한 learning direction의 부호를 가르는 일차 원리가 \(J\)에 있다는 강한 메시지를 제시한다. (근거: Section 8)

---
# Appendix A–N 확장 해설

이 절에서는 본문보다 더 세부적인 증명, 기하학적 해석, explicit solution, 하이퍼파라미터, noise injection 알고리즘, 데이터 샘플을 포함한 부록 전체를 순서대로 정리한다. 부록은 본문에서 선언된 결과의 엄밀한 근거를 제공하므로, 재현과 후속 연구를 위해서는 오히려 본문보다 중요할 수 있다. (근거: Appendix A–N)

---

## Appendix A. LLM as Multi-arm Bandit

### A.1 고전 MAB의 복습

[원문 근거] Appendix A.1은 먼저 고전적 multi-armed bandit 문제를 복습한다. \(K\)개의 arm이 있고 각 arm은 알려지지 않은 평균보상 \(\mu_a\)를 갖는다. 시간 \(T\) 동안 누적 regret은

$$
R_T = T\mu^\star - \sum_{t=1}^T \mu_{a_t},
\qquad \mu^\star = \max_a \mu_a
$$

로 정의된다. 저자들이 이 복습을 넣은 목적은 RLVR의 sequence-level decision을 bandit surrogate로 보는 논리적 배경을 제공하는 데 있다. noisy, delayed, partial feedback을 다루는 분석 도구로서 밴딧이 자연스럽기 때문이다. (근거: Appendix A.1)

### A.2 LLM completion space의 bandit화

[원문 근거] Appendix A.2는 본문 Section 4의 truncation argument를 자세히 기술한다. temperature가 0이 아니면 동일 프롬프트에서도 completion은 다르게 샘플링되며, 이론적으로 가능한 문자열 공간은 매우 크다. 그러나 실제 inference/training 환경에서는 \(L_{\max}\) 길이 제한이 존재하므로, 출력 공간은 유한 support처럼 취급할 수 있다. 이로써 LLM completion sampling은 categorical action over a very large but finite set로 해석된다. (근거: Appendix A.2)

### A.3 coarse-graining과 mode-level policy

[원문 근거] Appendix A.3는 coarse-graining map \(\phi\)를 통해 sequence를 reasoning mode로 묶는 정의를 formal하게 정리한다. 동일한 reasoning type을 공유하는 completion들을 하나의 arm으로 간주하면, induced categorical policy \(\pi_\theta\)가 생성되며, 이후 모든 mean-field analysis는 이 mode-level distribution에서 수행된다. 또한 good set \(H^+\), bad set \(H^-\) 분해와 bad mass \(p\)의 정의가 부록에서 다시 체계적으로 재정리된다. (근거: Appendix A.3)

---

## Appendix B. Noisy Rewards and Youden’s \(J\)

### B.1 기대 reward와 분산

[원문 근거] Appendix B는 noisy Bernoulli reward를 보다 체계적으로 정리한다. 기대 reward는

$$
q(p)=\mathbb E[r]=(1-\delta_{FN})-Jp
\tag{20}
$$

이고, 분산은

$$
\sigma^2(p)=\mathrm{Var}(r)=q(p)(1-q(p)).
\tag{21}
$$

같은 양을 전개식으로 쓰면

$$
(1-p)(1-\delta_{FN})\delta_{FN}
+
p\delta_{FP}(1-\delta_{FP})
+
p(1-p)J^2
\tag{22}
$$

가 된다. 이 식은 variance가 단지 Bernoulli noise 때문만이 아니라, good과 bad mixture 자체의 이질성에서도 기인함을 보여준다. (근거: Appendix B, Eq. 20–22)

### B.2 Figure 6: reward variance geometry

> [Figure 삽입 안내] 이 위치에 **논문 원문의 Figure 6** crop을 넣으십시오.  
> - 권장 캡션: `B.2 Figure 6: reward variance geometry`  
> - 편집 원칙: 도판 자체는 원문 이미지를 사용하고, 본문에서는 **이 도판이 무엇을 보여 주는지**만 해설합니다.

[원문 근거] Figure 6은 representative noise setting에서 \(\mathrm{Var}(r)\)와 \(p\)의 관계를 시각화하고, variance-maximizing \(p^\star\)와 최대 achievable variance \(\sigma_{\max}\)를 heatmap으로 보여준다. \(J>0\) informative region, \(J=0\) 경계, noise plane 상의 최대 분산 위치 등이 정리된다. 본문에서는 보상분산 \(\sigma(p)\)가 수렴 속도 식에 등장했기 때문에, Appendix B의 이 그림은 Section 3의 asymptotic law와 직접 연결된다. (근거: Figure 6, Appendix B)

### B.3 variance-maximizing bad mass

[원문 근거] variance는 \(q(p)=1/2\)일 때 최대가 되므로,

$$
p^\star = \mathrm{clip}\left(\frac{1/2-\delta_{FN}}{J},\,0,\,1\right)
$$

를 얻는다. 이는 “어떤 bad mass에서 verifier reward가 가장 noisy해지는가”를 알려주는 식이다. 대칭 노이즈와 비대칭 노이즈에서 \(p^\star\)의 위치가 달라지며, 이는 Section 3.3과 Appendix D의 intermediate learnability 논의와 맞물린다. (근거: Appendix B)

### B.4 z-score normalization의 conditional expectation

[원문 근거] group z-score reward를

$$
\tilde r = \frac{r-q(p)}{\sigma(p)}
\tag{23}
$$

로 정의하면,

$$
\mathbb E[\tilde r\mid good]=\frac{Jp}{\sigma(p)},
\qquad
\mathbb E[\tilde r\mid bad]= -\frac{J(1-p)}{\sigma(p)}
\tag{24}
$$

이며 전체 평균은

$$
\mathbb E[\tilde r]=0.
\tag{25}
$$

이 식이 본문 Section 2의 핵심인 good-bad gap \(J/\sigma(p)\)를 만든다. Remark B.1은 centered-only normalization을 쓰면 분모 \(\sigma(p)\)가 빠질 뿐 구조는 동일하다고 설명한다. Remark B.2는 reward alphabet를 \(\{\pm1\}\)로 바꾸더라도 선형 rescaling만 일어날 뿐 본질은 바뀌지 않는다고 덧붙인다. Remark B.3은 noise-free \(J=1\)에서 conditional expectation이 더 단순한 square-root 형태를 갖는다고 적는다. (근거: Appendix B, Eq. 23–25, Remarks B.1–B.3)

---

## Appendix C. Mean Field Dynamics

### C.1 REINFORCE gradient에서 mean-field drift로

[원문 근거] Proposition C.1은 REINFORCE gradient와 softmax Jacobian으로부터 expected parameter update가

$$
\mathbb E[\Delta \theta\mid p]=\eta J(p)A
$$

가 되고, 확률공간으로 pushforward하면

$$
\mathbb E[\Delta p\mid p]\approx \eta J(p)^2 A
$$

형태가 됨을 보인다. 이것이 Section 5의 natural-gradient 해석의 엄밀한 출발점이다. (근거: Proposition C.1)

### C.2 block symmetry와 good/bad specialization

[원문 근거] block symmetry 가정 아래 저자들은 good block의 모든 advantage가 \(a_g(p)\), bad block의 모든 advantage가 \(a_b(p)\)로 동일하다고 둔다.

$$
A_j=a_g(p),
\qquad
A_{b_m}=a_b(p),
\qquad
\Delta r(p)=a_b(p)-a_g(p).
\tag{27}
$$

noisy GRPO specialization에서는

$$
a_g(p)=\frac{Jp}{\sigma(p)},
\qquad
a_b(p)=-\frac{J(1-p)}{\sigma(p)},
\qquad
\Delta r(p)=-\frac{J}{\sigma(p)}.
\tag{32}
$$

를 얻는다. 이 결과를 이용해 total bad-mass drift

$$
\mathbb E[\Delta p]
=
-\eta\frac{J}{\sigma(p)}[p(1-p)]^2\big(\|y\|_2^2+\|z\|_2^2\big)
\tag{34}
$$

및 within-bad, within-good drift

$$
\mathbb E[\Delta z]
=
-\eta\frac{J}{\sigma(p)}p(1-p)\big(z\odot z - \|z\|_2^2 z\big)
\tag{35}
$$

$$
\mathbb E[\Delta y]
=
\eta\frac{J}{\sigma(p)}p(1-p)\big(y\odot y - \|y\|_2^2 y\big)
\tag{41}
$$

을 도출한다. 이는 본문 식 (12a)–(12c)의 discrete-time 원형이다. (근거: Appendix C.1–C.2)

### C.3 discrete-time에서 ODE로의 연결

[원문 근거] Appendix C.3은 discrete iteration을 continuous-time ODE로 연결한다. 한 iteration을 단위 시간으로 보는 경우,

$$
\dot p(t)=\eta(\mathrm{Diag}(p(t))-p(t)p(t)^\top)g(p(t))
\tag{42}
$$

를 얻고, time-rescaled limit \(t=\eta t_{\mathrm{discrete}}\)를 사용하면

$$
\frac{dp}{dt}=(\mathrm{Diag}(p)-pp^\top)g(p)
\tag{44}
$$

가 된다. Remark C.4는 두 표기는 time scale만 다를 뿐 궤적은 동일하며, 본 논문은 \(\eta\) 의존성을 명시적으로 보이기 위해 unit-time notation을 유지한다고 설명한다. (근거: Appendix C.3)

---

## Appendix D. Maximal Learnability

### D.1 instantaneous progress의 정의

[원문 근거] Appendix D는 Section 3.3의 직관을 수식으로 확장한다. instantaneous bad-mass reduction 속도를

$$
|\Delta p| \propto \Delta(p)[p(1-p)]^2,
\qquad
\Delta(p)=\mathbb E[\tilde r\mid good]-\mathbb E[\tilde r\mid bad]
\tag{45}
$$

로 두고, z-score normalization 하에서는

$$
\Delta(p)=\frac{J}{\sigma(p)}
\tag{46}
$$

이므로 learnability speed를

$$
L(p;\delta_{FN},\delta_{FP})
=
\frac{J}{\sigma(p)}[p(1-p)]^2
\tag{47}
$$

로 정의한다. (근거: Appendix D, Eq. 45–47)

### D.2 noise-free와 symmetric noise의 경우

[원문 근거] noise-free에서는

$$
L(p;0,0)=[p(1-p)]^{3/2}
\tag{48}
$$

이므로 \(p^\dagger=1/2\)에서 최대가 된다. symmetric noise \((\delta_{FN}=\delta_{FP}=\delta)\)에서는 최적점이 여전히 \(1/2\)에 남고,

$$
L_{\max}(\delta,\delta)=\frac{1-2\delta}{8}
\tag{50}
$$

로 peak 높이만 감소한다. (근거: Appendix D, Eq. 48–50)

### D.3 asymmetric noise와 Cardano 언급

[원문 근거] asymmetric noise에서는 stationary condition이

$$
2\frac{1-2p}{p(1-p)}
+
\frac{J}{2}\frac{1-2q(p)}{q(p)(1-q(p))}=0
\tag{51}
$$

이 되어 interior maximizer가 이동한다. 일반적으로 cubic equation이 되므로 Cardano formula로 닫힌 형태를 얻을 수 있으나, 실질적 계산은 수치적으로 수행하면 충분하다고 저자들은 적는다. Figure 7은 전체 noise plane에 걸쳐 \(p^\dagger\)와 \(L_{\max}\)를 시각화하며, 대칭 노이즈는 최적 난이도 위치를 보존하고 비대칭 노이즈는 이를 이동시킨다는 점을 보여준다. (근거: Appendix D, Eq. 51, Figure 7)

---

## Appendix E. Lyapunov Analysis and the Role of \(J\)

### E.1 Lyapunov 함수의 구성

[원문 근거] Appendix E는 sign-of-\(J\) 결과를 Lyapunov 분석으로 다시 증명한다. good mass를 \(s(p)\), bad mass를 \(P_{bad}=1-s\)로 두고 potential \(F\)를 \(F'(s)=\Delta(s)\)로 정의하면,

$$
\frac{d}{dt}F(p(t))
=
\eta\|J(p(t))A(p(t))\|_2^2 \ge 0
\tag{57}
$$

가 된다. 즉 \(F\)는 시간에 따라 비감소하는 Lyapunov 함수다. (근거: Appendix E, Eq. 57)

### E.2 good mass의 단조성

[원문 근거] normalized coordinates에서는

$$
\dot s
=
\eta\Delta(s)[s(1-s)]^2(\|y\|_2^2+\|z\|_2^2)
\tag{59}
$$

를 얻는다. \(\Delta(s)\)의 부호가 곧 \(J\)의 부호와 같으므로, \(J>0\)이면 \(s\uparrow 1\), \(J<0\)이면 \(s\downarrow 0\)가 된다. \(J=0\)이면 모든 점이 평형이다. 저자들은 이를 \(J=0\)에서의 **exchange of stability**로 해석한다. 즉 임계점에서 안정한 면(face)이 good face에서 bad face로 뒤바뀐다. (근거: Appendix E, Eq. 59)

### E.3 정량적 tail bound

[원문 근거] Appendix E는 Section 3에서 서술한 \(O(1/t)\) tail bound를 Lyapunov 관점에서도 재정리한다. 따라서 rate law는 단순한 heuristic이 아니라 독립적인 안정성 분석과도 일관된다는 점이 확인된다. (근거: Appendix E)

---

## Appendix F. Importance Sampling and PPO Clipping

### F.1 exact ratio의 전개

[원문 근거] Appendix F는 practical GRPO의 importance sampling과 clipping 효과를 엄밀히 분석한다. ratio를

$$
\rho_i = \frac{p_i^+}{p_i}
$$

로 두고 small-step \(\|p^+-p\|=O(\eta)\)를 사용하면, \(\rho_i=1+O(\eta)\)다. 이를 clipped surrogate에 대입해 전개하면 extra terms가 생기지만, softmax pushforward 후의 leading-order mean update에는 영향을 주지 않는다는 것이 부록의 핵심 주장이다. (근거: Appendix F)

### F.2 phase portrait의 보존

[원문 근거] Appendix F의 결론은 간결하다. PPO clipping과 importance sampling correction은 small-step regime에서 \(O(\eta^2)\) 수준의 보정에 머물며, 본 논문이 다루는 sign-of-\(J\) phase transition은 여전히 유지된다. 따라서 “실전 알고리즘은 본문 이론과 다르다”는 식의 반론은, 적어도 1차 mean-field 수준에서는 성립하지 않는다. (근거: Appendix F 결론)

---

## Appendix G. KL Regularization

### G.1 reverse-KL의 two-class 및 full decomposition

[원문 근거] reference policy를

$$
p^{ref} = ((1-p_{ref})y^{ref},\; p_{ref}z^{ref})
$$

로 두면, reverse-KL은 정확히

$$
D_{KL}(p\|p^{ref})
=
\underbrace{p\log\frac{p}{p_{ref}} + (1-p)\log\frac{1-p}{1-p_{ref}}}_{\text{two-class term}}
+
(1-p)D_{KL}(y\|y^{ref})
+
pD_{KL}(z\|z^{ref})
$$

로 분해된다. 이 분해는 aggregate bad mass와 within-block mismatch를 분리하는 핵심 도구다. (근거: Lemma G.1)

### G.2 two-class KL term의 효과

[원문 근거] two-class KL penalty는 \(p\)에만 작용하며, \(y\)와 \(z\)에는 deterministic drift를 주지 않는다. 결과적으로

$$
\dot p_{KL,2c}
=
-\beta p(1-p)(\ell-\ell_{ref}),
\qquad
\ell=\log\frac{p}{1-p}
$$

를 얻는다. 이는 bad mass logit을 reference logit 쪽으로 되돌리는 restoring force다. (근거: Proposition G.2)

### G.3 full reverse-KL의 richer dynamics

[원문 근거] full reverse-KL에서는

$$
\dot p_{KL,full}
=
-\beta p(1-p)
\Big(
\ell-\ell_{ref}-D_{KL}(y\|y^{ref})+D_{KL}(z\|z^{ref})
\Big)
$$

를 얻는다. 즉 good block mismatch와 bad block mismatch가 서로 다른 부호로 aggregate bad mass에 개입한다. 또한 \(y\)와 \(z\) 자체도 각각 reference block 내부 분포 쪽으로 끌려가는 within-block pull을 갖는다. (근거: Proposition G.3, Remark G.4)

### G.4 reward drift와 결합된 전체 식

[원문 근거] reward-driven drift와 KL term을 합치면 two-class 관점에서는

$$
\dot p
=
-\eta\frac{J}{\sigma(p)}[p(1-p)]^2(s_2+t_2)
-
\beta p(1-p)(\ell-\ell_{ref})
\tag{68}
$$

을 얻고, full reverse-KL에서는 이에 대응하는 식 (69)가 성립한다. Theorem G.6은 \(\beta>0\)이면 unique globally stable interior equilibrium이 존재함을 보이고, Corollary G.7은 특히 \(J<0\)에서도 \(\beta>0\)가 total collapse를 방지한다고 정리한다. (근거: Appendix G, Eq. 68–69, Theorem G.6, Corollary G.7)

---

## Appendix H. Properties of the Simplex

### H.1 softmax Jacobian의 선형대수적 성질

[원문 근거] Appendix H는 simplex 기하의 기본 성질을 정리한다. \(\Delta^{d-1}\)와 tangent space \(T_p\Delta\)를 정의하고, softmax Jacobian

$$
J(p)=\mathrm{Diag}(p)-pp^\top
$$

이 대칭이고 positive semidefinite이며, null space가 \(\mathrm{span}\{\mathbf{1}\}\), image가 접공간이라는 점을 확인한다. 이는 \(J(p)\)가 확률단체 위 natural projector라는 사실을 의미한다. (근거: Appendix H)

### H.2 Shahshahani metric과 KL local expansion

[원문 근거] Appendix H는 Shahshahani metric을

$$
\langle u,v\rangle_{\mathrm{Shah};p}=\sum_i \frac{u_i v_i}{p_i}
\tag{73}
$$

로 정의한다. negative entropy의 Hessian이 바로 이 metric tensor와 연결되며, forward KL은 local quadratic form으로

$$
D_{KL}(p+\delta\|p)
=
\frac12\delta^\top \mathrm{Diag}(p)^{-1}\delta + o(\|\delta\|^2)
\tag{76}
$$

를 가진다. 이 결과는 Figure 4의 기하학적 설명을 엄밀히 뒷받침한다. (근거: Appendix H, Eq. 73, 76)

### H.3 mirror ascent와 natural gradient의 일치

[원문 근거] Proposition H.7과 Corollary H.8은 entropic mirror-ascent step이 1차 근사에서 Shahshahani natural-gradient flow의 Euler step과 같음을 증명한다. 따라서 “GRPO mean-field ODE”, “replicator dynamics”, “mirror ascent”가 서로 다른 언어일 뿐, 동일한 기하적 구조를 공유한다는 본문 Section 5.2의 해석이 엄밀히 정당화된다. (근거: Appendix H, Proposition H.7, Corollary H.8)

---
## Appendix I. Inner Dynamics of the Good Arms

### I.1 good block의 autonomous dynamics

[원문 근거] Appendix I는 good block \(y\)만 떼어내 상세히 분석한다. 먼저

$$
y_j = \frac{p_j}{1-p}=\mathrm{softmax}(\theta_{good})_j
\tag{80}
$$

로 정의되므로, \(y\)는 bad logits와는 무관한 내부 good-coordinate라는 점이 분명해진다. 기대 drift는

$$
\mathbb E[\Delta y]
=
\kappa(p)\big(y\odot y - \|y\|_2^2 y\big),
\qquad
\kappa(p)=-\eta p(1-p)\Delta r(p)
\tag{82,83}
$$

이며 noisy GRPO specialization에서는

$$
\kappa(p)=\eta \frac{J}{\sigma(p)}p(1-p)
$$

가 된다. (근거: Appendix I)

[원문 근거] Appendix I의 footnote 2는 중요한 비교를 제시한다. 만약 알고리즘이 pure natural-gradient replicator \(\dot p=\eta J(p)A\)였다면, block symmetry 아래 good 내부 \(y\)에는 deterministic drift가 사라지고 sampling noise만 남을 수 있다. 즉 본 논문이 밝히는 good-block collapse는 “replicator dynamics 일반”의 필연이라기보다, GRPO의 구체적 구조와 softmax pushforward가 결합해 생기는 결과라는 점이 드러난다. (근거: Appendix I, footnote 2)

### I.2 internal time과 Lyapunov 구조

[원문 근거] internal time \(\tau\)를 도입하면 good-block dynamics는 autonomous ODE

$$
\frac{dy}{d\tau}=y\odot y - s_2 y,
\qquad s_2=\|y\|_2^2
\tag{84}
$$

로 바뀐다. 이 ODE는 strict Lyapunov 함수를 가지며, \(\kappa\ge 0\)이면 \(s_2\)가 단조 증가한다. fixed point는 support-uniform point들이고, \(\kappa>0\)이면 vertex가 attractor, full-uniform point는 saddle 성격을 갖는다. footnote 3은 \(K=2\)와 one-vs-rest 대칭 slice에서 explicit closed form도 제시한다. (근거: Appendix I, Eq. 84, footnote 3)

### I.3 collision term과 logit envelope

[원문 근거] Appendix I.1은 collision term \(s_2\)의 진화를 정리한다. 이를 바탕으로 bad-mass logit에 대해

$$
\frac{p_0}{1-p_0}e^{-2\tau}
\le
\frac{p(\tau)}{1-p(\tau)}
\le
\frac{p_0}{1-p_0}e^{-(1/K+1/M)\tau}
$$

와 같은 지수형 envelope가 얻어진다. 즉 internal time에서 bad mass logit은 최소 \(1/K+1/M\), 최대 2의 기울기로 감소한다. (근거: Appendix I.1)

### I.4 near-uniform 초기조건과 미세 비대칭

[원문 근거] Theorem I.7은 near-uniform initialization에서 작은 이질성 \(\zeta_0,\xi_0\)가 logit trajectory를 어떻게 수정하는지 정량화한다. Corollary I.8은 small-\(\tau\) 영역에서 finer details보다 \(\zeta_0\), \(\xi_0\) 두 스칼라가 1차적으로 중요하다는 결론을 준다. 이는 본문 Section 6.2.2와 직접 연결된다. (근거: Appendix I, Theorem I.7, Corollary I.8)

### I.5 generic initial condition과 winner selection

[원문 근거] Theorem I.11은 좌표 tie가 없는 generic initial condition에서 trajectory가 초기 최대 good arm이 정의하는 vertex로 수렴한다고 증명한다. Proposition I.12는 non-winning coordinate가 internal time에서 \(e^{-\tau}\)로 사라지는 exponential polarization을 제시한다. Proposition I.15, I.16, Corollary I.17과 Table 2는 안정성 구조를 요약한다.

- uniform point는 \(J>0\)에서 unstable, \(J<0\)에서 stable
- vertex는 \(J>0\)에서 stable, \(J<0\)에서 unstable

즉 good block은 learning regime에서 본질적으로 한 mode로 polarize된다. (근거: Appendix I.2–I.3, Table 2)

### I.6 internal time에서 physical time으로

[원문 근거] Theorem I.19는 internal time 결과를 physical time으로 옮긴다. attractor 근처에서 \(\sigma(p)\sim\sigma_0 p^\gamma\)라고 두면, \(J>0\) branch에서

- \(\gamma<1\): \(p(t)\asymp t^{-1/(1-\gamma)}\)
- \(\gamma=1\): \(p(t)\asymp e^{-(aJ/\sigma_0)t}\), \(a=1+1/M\)
- \(\gamma>1\): finite-time absorption law

가 성립한다. Corollary I.20은 “누가 이기는가”에 대한 답이 초기 최대 good arm이라는 점을, Corollary I.21은 \(p(\tau)=\Theta(e^{-(1+1/M)\tau})\) sharp envelope를 제시한다. (근거: Appendix I.4, Corollaries I.20–I.21)

### I.7 explicit solution machinery

[원문 근거] Appendix I.5–I.6은 auxiliary scalar \(I(\tau)\), moment sum \(M_r(I)\), integral map \(\tau(I)\) 등을 도입해 \(y_j(\tau)\), \(s_2(\tau)\), \(S(\tau)=\int s_2\)를 닫힌 형태 또는 implicit form으로 적는다. 이는 내부 good dynamics를 수치 적분 없이도 정밀히 분석할 수 있게 해주는 기술적 장치다. (근거: Appendix I.5–I.6)

---

## Appendix J. Inner Dynamics of the Bad Arms

### J.1 정의와 기본 drift

[원문 근거] Appendix J는 Appendix I의 부호 반전 버전이다. bad-block normalized coordinate를

$$
z_m = \frac{p_{b_m}}{p}=\mathrm{softmax}(\theta_{bad})_m
\tag{110}
$$

로 정의하고, drift를

$$
\mathbb E[\Delta z]
=
-\kappa(p)\big(z\odot z - \|z\|_2^2 z\big)
\tag{113}
$$

로 적는다. 즉 \(J>0\)이면 bad block은 균등화되고, \(J<0\)이면 bad block 내부에서도 winner-take-all collapse가 발생한다. (근거: Appendix J)

### J.2 장기 거동과 안정성

[원문 근거] Theorem J.6은 interior initialization에서 다음을 보인다.

- \(J>0\): \(z(t)\to \frac1M\mathbf{1}\), 따라서 \(t_2\to 1/M\)
- \(J<0\): 초기 최대 bad arm이 정의하는 vertex로 수렴

Table 3은 안정성 구조가 good block과 정확히 부호 반전된다는 점을 요약한다. 즉 learning regime에서는 bad arm들이 diffuse되고, anti-learning regime에서는 오히려 bad arm 내부에서도 하나의 잘못된 모드가 지배적이 된다. (근거: Appendix J, Table 3)

### J.3 logit equation에 대한 영향

[원문 근거] Remark J.9는 multi-bad model에서

$$
\frac{d}{d\tau}\log\frac{p}{1-p}=-(\|y\|_2^2+\|z\|_2^2)
$$

가 유지되므로, \(\|z\|_2^2\)는 결국 bounded multiplicative factor로 bad-mass drift에 개입한다고 설명한다. 즉 bad block의 내부 구조는 속도를 바꾸지만, \(J\)의 sign이 정하는 전체 방향성을 뒤집지는 못한다. (근거: Appendix J, Remark J.9)

---

## Appendix K. Shahshahani Geometry and the Within-Good Flow

### K.1 within-good flow의 기하학적 재해석

[원문 근거] Appendix K는 good-block dynamics를 Shahshahani geometry 언어로 다시 쓴다. Section 5와 동일하게

$$
\dot y = \kappa(p)\big(y\odot y - \|y\|_2^2 y\big)
$$

를 두고, potential

$$
\Phi(y)=\frac12\|y\|_2^2
$$

에 대해

$$
\dot y = \kappa(p)\,\mathrm{grad}_{Shah}\Phi(y)
$$

임을 Proposition K.1이 증명한다. 즉 good-block winner-take-all은 Fisher/Shahshahani 기하 위에서 concentration potential을 오르는 자연그래디언트 상승으로 이해된다. (근거: Appendix K, Proposition K.1)

### K.2 \(\Phi\)의 단조성과 안정성

[원문 근거] Corollary K.2는

$$
\frac{d}{dt}\Phi(y)=\kappa(p)\,\mathrm{Var}_{i\sim y}(y_i)\ge 0
$$

를 제시한다. \(\kappa>0\)이면 concentration potential이 증가하므로, 균등분포는 불안정하고 vertex가 안정적이다. Proposition K.3은 equilibrium이 face barycenter들이며, \(\kappa>0\)일 때 vertex만 안정하고 \(\kappa<0\)일 때는 full-uniform point만 안정하다고 정리한다. (근거: Appendix K)

---

## Appendix L. Hyperparameters and Training Details

[원문 근거] Table 4는 실험 재현의 핵심 하이퍼파라미터를 제공한다.

- Base model: Qwen2.5-3B
- Global batch size: 16
- Train steps: 1410
- Epochs: 2
- Rollout num per prompt: 8
- Rollout temperature: 1.0
- Top-p: 1.0
- Top-k: -1
- Max prompt length: 4000
- Max response length: 4000
- PPO mini-batch size: 32
- Advantage type: GRPO
- Clipping thresholds: \(\epsilon_{low}=\epsilon_{high}=0.2\)
- Optimizer: Adam
- Learning rate: \(10^{-6}\)
- Weight decay: 0.1
- \((\beta_1,\beta_2)=(0.9,0.999)\)
- Gradient norm clip: 1.0
- LR schedule: constant
- Warmup: 10 steps
- KL coefficient: 0.0
- Evaluation decoding temperature: 0.0 (greedy)

GPU 종류, mixed precision 여부, wall-clock training time 등은 별도로 명시되어 있지 않다. (근거: Appendix L, Table 4)

---

## Appendix M. Noise Injection Pseudocode

[원문 근거] Algorithm 1은 noisy verifier wrapper를 의사코드 수준에서 명료하게 제시한다. Oracle(program)로부터 ground-truth correctness \(z\in\{0,1\}\)를 얻은 뒤,

- \(z=1\)이면 \(\mathrm{Bernoulli}(\mathrm{TPR})\)를 샘플링
- \(z=0\)이면 \(\mathrm{Bernoulli}(\mathrm{FPR})\)를 샘플링

하여 reward \(r\)를 반환한다. 즉 실험의 noise model은 매우 단순하고 투명한 Bernoulli corruption wrapper다. (근거: Appendix M, Algorithm 1)

---

## Appendix N. Data Sample

[원문 근거] Appendix N은 실제 데이터 샘플을 보여준다. 제시된 예시는 **kMarsh** 문제로, marsh가 포함된 직사각형 격자에서 가장 큰 직사각형 fence perimeter를 구하는 과제다. 함수 시그니처 `kMarsh(grid)`, 입력 형식, 제약 조건 \(2\le m,n\le 500\), 출력 형식, 샘플 입출력, 그리고 JSON 형태의 test cases가 포함된다. 이 부록은 실험 데이터가 추상적 개념 검증용 toy problem이 아니라, 실제 코드 생성 benchmark의 구체 문제라는 점을 보여준다. (근거: Appendix N)

---

# 종합 정리

이 논문은 RLVR의 noisy verifier 문제를 극도로 간결한 형태로 압축한다. 핵심 질문은 verifier가 평균적으로 정답을 더 자주 칭찬하는가, 아니면 오답을 더 자주 칭찬하는가이다. 그 대답은 Youden’s index

$$
J=\mathrm{TPR}-\mathrm{FPR}
$$

의 부호로 표현된다. \(J>0\)이면 bad-mode mass는 감소하고 learning이 일어나며, noise는 주로 수렴 속도에 개입한다. \(J=0\)이면 deterministic drift가 사라져 학습 방향성이 없어진다. \(J<0\)이면 dynamics가 반전되어 anti-learning과 collapse가 발생한다. (근거: Abstract, Sections 3, 8)

동시에 이 논문은 RLVR의 더 미묘한 구조도 드러낸다. completion-level RL은 natural하게 bandit으로 coarse-grain될 수 있고, 그 dynamics는 확률단체 위 natural-gradient replicator flow로 해석된다. good mode 내부에서는 winner-take-all이 발생하고, bad mode 내부에서는 learning regime에서 entropy-increasing spread가 나타난다. finite rollout은 genetic drift를 유발하고, KL regularization은 collapse를 부드럽게 완화하지만 verifier의 근본적 판별력 부족을 구제하지는 못한다. (근거: Sections 4–6, Appendices H–K)

연구적 의미는 분명하다. verifier noise는 “학습 품질의 부차적 저하 요인”이 아니라, **학습 방향과 안정성의 일차적 결정요인**이다. 실무적으로는 verifier의 \(J\)를 계량하지 않은 채 RLVR를 확장하는 것이 위험할 수 있음을 시사하며, 이론적으로는 sequence-level RLVR를 probability-simplex geometry와 mean-field ODE로 분석하는 강력한 틀을 제공한다. (근거: Sections 7–8)
