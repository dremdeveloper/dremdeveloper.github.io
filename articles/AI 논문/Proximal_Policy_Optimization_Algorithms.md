# Proximal Policy Optimization Algorithms

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov  
OpenAI

## Abstract

이 논문은 강화학습을 위한 새로운 policy gradient 방법군을 제안한다. 제안하는 방법은 환경과 상호작용하면서 데이터를 수집하고, 그다음 수집된 데이터에 대해 surrogate objective를 stochastic gradient ascent로 최적화하는 과정을 번갈아 수행한다. 기존의 표준 policy gradient 방법은 데이터 샘플 하나당 한 번의 gradient update를 수행하는 경우가 많지만, 이 논문은 하나의 batch에 대해 여러 번의 minibatch update를 수행할 수 있게 해 주는 새로운 objective를 제시한다.

저자들은 이 방법군을 proximal policy optimization, 즉 PPO라고 부른다. PPO는 trust region policy optimization(TRPO)의 장점을 상당 부분 유지하면서도 구현이 훨씬 단순하고, 더 일반적인 구조에 적용할 수 있으며, 경험적으로 더 나은 sample complexity를 보인다고 주장한다. 실험은 시뮬레이션된 로봇 보행 과제와 Atari 게임을 포함한 benchmark에서 수행되었고, 그 결과 PPO는 기존의 여러 online policy gradient 방법보다 우수한 성능을 보이며, sample complexity·단순성·wall-time 사이에서 균형이 잘 잡힌 방법이라는 결론을 제시한다.

## 1. Introduction

최근에는 신경망 함수 근사기를 사용하는 강화학습에서 여러 접근이 경쟁적으로 발전해 왔다. 대표적인 방법으로는 deep Q-learning, vanilla policy gradient 방법, 그리고 trust region 또는 natural policy gradient 계열 방법이 있다. 그러나 저자들은 아직도 개선의 여지가 크다고 본다. 이상적인 방법은 큰 모델과 병렬 구현으로 쉽게 확장될 수 있어야 하고, 데이터를 효율적으로 사용해야 하며, 하이퍼파라미터를 세밀하게 조정하지 않아도 다양한 문제에서 안정적으로 동작해야 한다.

각 계열의 한계도 분명하다. 함수 근사기를 사용하는 Q-learning은 단순한 문제에서도 실패하는 경우가 많고, 이론적 이해도 충분하지 않다. Vanilla policy gradient는 데이터 효율성과 안정성이 떨어진다. TRPO는 안정성과 성능 면에서는 강력하지만 알고리즘 자체가 상대적으로 복잡하고, dropout처럼 내부에 노이즈가 포함된 구조나, policy와 value function 사이의 파라미터 공유 구조, 혹은 auxiliary task를 함께 붙이는 구조와 잘 맞지 않는다.[^1]

이 논문의 목적은 TRPO의 데이터 효율성과 안정적인 성능을 최대한 유지하면서도, second-order 기법 없이 first-order optimization만으로 구현할 수 있는 알고리즘을 제시하는 데 있다. 이를 위해 저자들은 clipped probability ratio를 사용하는 새로운 objective를 제안한다. 이 objective는 정책 성능에 대한 비관적 추정치, 즉 lower bound를 형성하도록 설계되어 있으며, 실제 학습은 현재 policy로 데이터를 수집한 뒤 그 데이터에 대해 여러 epoch의 optimization을 수행하는 방식으로 진행된다.

실험에서는 surrogate objective의 여러 변형을 비교하여 clipped probability ratio를 사용하는 버전이 가장 잘 작동함을 보이고, 그다음 PPO를 기존 알고리즘들과 비교한다. 연속 제어 과제에서는 PPO가 비교 대상보다 더 좋은 성능을 보이고, Atari 환경에서는 A2C보다 sample complexity 측면에서 훨씬 우수하며, ACER와는 비슷한 수준의 성능을 더 단순한 방식으로 달성한다.

## 2. Background: Policy Optimization

### 2.1 Policy Gradient Methods

Policy gradient 방법은 정책의 gradient estimator를 계산하고, 그 추정량을 stochastic gradient ascent 알고리즘에 넣어 policy를 직접 최적화하는 방식이다. 가장 널리 사용되는 gradient estimator는 다음과 같다.

$$
\hat g
=
\hat{\mathbb E}_t
\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\hat A_t
\right]
\tag{1}
$$

여기서 $\pi_\theta$는 확률적 정책이고, $\hat A_t$는 시점 $t$에서의 advantage function 추정량이다. $\hat{\mathbb E}_t[\cdot]$는 샘플링과 최적화를 번갈아 수행하는 알고리즘 안에서, 유한한 batch에 대해 계산한 empirical average를 뜻한다. 즉 식 (1)은 현재 policy 아래에서 관측된 행동의 로그확률을 어느 방향으로 얼마나 바꿀지를 advantage로 가중해 평균낸 gradient라고 볼 수 있다.

자동미분 소프트웨어를 사용하는 실제 구현에서는 gradient estimator를 직접 적기보다, 그 gradient가 정확히 식 (1)이 되도록 objective를 구성하는 것이 일반적이다. 그 objective는 다음과 같다.

$$
L^{PG}(\theta)
=
\hat{\mathbb E}_t
\left[
\log \pi_\theta(a_t \mid s_t)\,\hat A_t
\right]
\tag{2}
$$

식 (2)를 그대로 두고 동일한 trajectory에 대해 여러 번 optimization step을 수행하면 좋아 보일 수 있다. 그러나 저자들은 이런 방식이 이론적으로 충분히 정당화되지 않고, 실제로는 policy가 한 번에 지나치게 크게 변하는 파괴적인 업데이트를 자주 일으킨다고 지적한다. Section 6.1에서 보게 되듯이, clipping이나 penalty 없이 같은 데이터를 여러 번 재사용하는 설정은 실제 성능도 좋지 않다.

### 2.2 Trust Region Methods

TRPO에서는 정책 업데이트의 크기를 제한하는 제약식 아래에서 surrogate objective를 최대화한다. 구체적인 문제는 다음과 같다.

$$
\max_\theta\;
\hat{\mathbb E}_t
\left[
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
\hat A_t
\right]
\tag{3}
$$

subject to

$$
\hat{\mathbb E}_t
\left[
KL\!\left[
\pi_{\theta_{\mathrm{old}}}(\cdot \mid s_t),
\pi_\theta(\cdot \mid s_t)
\right]
\right]
\le \delta
\tag{4}
$$

여기서 $\theta_{\mathrm{old}}$는 업데이트 이전 정책의 파라미터 벡터를 뜻한다. 식 (3)의 목적함수는 old policy에서 수집한 데이터를 사용하면서도 new policy의 성능 변화를 근사하기 위한 surrogate이고, 식 (4)는 새로운 policy가 old policy에서 너무 멀리 벗어나지 않도록 평균 KL divergence를 제한하는 trust region 역할을 한다. TRPO는 objective를 선형 근사하고 constraint를 이차 근사한 뒤 conjugate gradient 알고리즘을 사용해 이 문제를 효율적으로 근사 해결한다.

TRPO를 정당화하는 이론은 hard constraint 대신 penalty를 사용하는 형태도 시사한다. 즉 다음과 같은 unconstrained optimization을 생각할 수 있다.

$$
\max_\theta\;
\hat{\mathbb E}_t
\left[
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
\hat A_t
-
\beta\,
KL\!\left[
\pi_{\theta_{\mathrm{old}}}(\cdot \mid s_t),
\pi_\theta(\cdot \mid s_t)
\right]
\right]
\tag{5}
$$

여기서 $\beta$는 penalty coefficient다. 이런 형태가 가능한 이유는, 특정 surrogate objective가 상태별 최대 KL을 사용했을 때 policy 성능에 대한 lower bound, 즉 비관적 경계를 이룬다는 사실 때문이다. 하지만 TRPO가 실제로 hard constraint를 채택한 이유도 분명하다. 하나의 고정된 $\beta$ 값이 서로 다른 여러 문제에 동시에 잘 맞기 어렵고, 같은 문제 안에서도 학습이 진행되면서 적절한 값이 달라질 수 있기 때문이다. 따라서 TRPO처럼 단조로운 성능 향상을 흉내 내는 first-order 알고리즘을 만들기 위해서는, 식 (5)를 고정된 계수로 SGD에 넣는 것만으로는 충분하지 않다.

## 3. Clipped Surrogate Objective

이제 저자들은 PPO의 핵심이 되는 clipped surrogate objective를 도입한다. 먼저 다음의 확률비를 정의한다.

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
$$

따라서 $\theta = \theta_{\mathrm{old}}$이면 항상 $r_t(\theta)=1$이다. TRPO가 최대화하는 surrogate objective는 다음처럼 다시 쓸 수 있다.

$$
L^{CPI}(\theta)
=
\hat{\mathbb E}_t
\left[
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
\hat A_t
\right]
=
\hat{\mathbb E}_t
\left[
r_t(\theta)\,\hat A_t
\right]
\tag{6}
$$

여기서 위첨자 $CPI$는 conservative policy iteration에서 온 것이다. 문제는 이 objective를 제약식 없이 그대로 최대화하면 policy가 지나치게 크게 바뀔 수 있다는 점이다. 따라서 저자들은 $r_t(\theta)$가 1에서 멀어질수록 그 변화에 불이익을 주는 방향으로 objective를 수정한다.

논문이 제안하는 핵심 objective는 다음과 같다.

$$
L^{CLIP}(\theta)
=
\hat{\mathbb E}_t
\left[
\min\!\left(
r_t(\theta)\,\hat A_t,\;
\operatorname{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\,\hat A_t
\right)
\right]
\tag{7}
$$

여기서 $\epsilon$은 하이퍼파라미터이며, 논문에서는 예시로 $\epsilon=0.2$를 든다. 식 (7)의 동기는 다음과 같다. $\min$ 내부의 첫 번째 항은 원래의 $L^{CPI}$ 항이고, 두 번째 항은 확률비를 $[1-\epsilon,\;1+\epsilon]$ 구간으로 잘라낸 항이다. 이 clipped term은 $r_t$가 그 구간 밖으로 더 멀리 움직여도 objective가 계속 좋아지지 않도록 만든다. 마지막에 둘 중 작은 값을 취함으로써 최종 objective는 unclipped objective에 대한 lower bound, 즉 비관적 경계가 된다.

이 구조의 중요한 점은, 확률비의 변화가 objective를 더 좋게 만들 때에만 그 개선을 무시하고, objective를 더 나쁘게 만드는 변화는 그대로 반영한다는 것이다. 따라서 $L^{CLIP}(\theta)$는 $\theta_{\mathrm{old}}$ 근방, 즉 $r=1$ 부근에서는 $L^{CPI}(\theta)$와 1차 근사 수준에서 동일하게 동작하지만, 정책이 old policy에서 멀어질수록 둘의 값은 달라진다.

Figure 1은 $L^{CLIP}$을 구성하는 단일 timestep 항이 확률비 $r$에 따라 어떻게 변하는지를 보여 준다. Advantage가 양수일 때는 좋은 행동의 확률을 지나치게 키우는 방향의 이득이 일정 지점 이후 더 이상 커지지 않고, advantage가 음수일 때는 나쁜 행동의 확률을 지나치게 줄이는 방향의 이득이 마찬가지로 제한된다.

<figure>
  <img src="/assets/images/Proximal_Policy_Optimization_Algorithms/01_Figure_1.png" alt="Figure 1. Clipped surrogate objective as a function of the probability ratio for positive and negative advantages." />
  <figcaption><strong>Figure 1.</strong> Clipped surrogate objective as a function of the probability ratio for positive and negative advantages.</figcaption>
</figure>

Figure 2는 같은 아이디어를 실제 policy update 방향 위에서 보여 준다. 저자들은 연속 제어 문제에서 PPO로 얻은 정책 업데이트 방향을 따라 초기 파라미터 $\theta_{\mathrm{old}}$와 업데이트된 파라미터를 선형 보간하면서 여러 objective가 어떻게 변하는지 비교한다. 이 그림을 통해 $L^{CLIP}$이 $L^{CPI}$의 lower bound 역할을 하며, 지나치게 큰 policy update에 대해서는 사실상 penalty처럼 작동한다는 점을 직관적으로 확인할 수 있다.

<figure>
  <img src="/assets/images/Proximal_Policy_Optimization_Algorithms/02_Figure_2.png" alt="Figure 2. Surrogate objectives and KL along the policy update direction on Hopper-v1." />
  <figcaption><strong>Figure 2.</strong> Surrogate objectives and KL divergence along the policy update direction on Hopper-v1.</figcaption>
</figure>

## 4. Adaptive KL Penalty Coefficient

Clipped surrogate objective와는 별도로, 저자들은 adaptive KL penalty를 사용하는 대안도 제시한다. 이 방식은 clipped objective를 대신할 수도 있고, 경우에 따라서는 함께 사용할 수도 있다. 핵심 아이디어는 KL divergence에 penalty를 두되, 각 policy update마다 실제 KL이 목표값 $d_{\mathrm{targ}}$ 근처가 되도록 penalty coefficient를 자동 조정하는 것이다.

저자들은 이 방법이 실험에서는 clipped surrogate objective보다 성능이 떨어졌다고 밝히지만, 중요한 baseline이기 때문에 논문에 포함한다. 가장 단순한 형태의 알고리즘은 각 policy update에서 다음 절차를 수행한다.

첫째, 여러 epoch의 minibatch SGD를 사용해 다음의 KL-penalized objective를 최적화한다.

$$
L^{KLPEN}(\theta)
=
\hat{\mathbb E}_t
\left[
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
\hat A_t
-
\beta\,
KL\!\left[
\pi_{\theta_{\mathrm{old}}}(\cdot \mid s_t),
\pi_\theta(\cdot \mid s_t)
\right]
\right]
\tag{8}
$$

둘째, 실제 KL divergence의 평균을 다음처럼 계산한다.

$$
d
=
\hat{\mathbb E}_t
\left[
KL\!\left[
\pi_{\theta_{\mathrm{old}}}(\cdot \mid s_t),
\pi_\theta(\cdot \mid s_t)
\right]
\right]
$$

셋째, 계산된 $d$가 목표값에서 얼마나 벗어났는지에 따라 $\beta$를 조정한다.

- $d < d_{\mathrm{targ}}/1.5$ 이면 $\beta \leftarrow \beta/2$
- $d > d_{\mathrm{targ}}\times 1.5$ 이면 $\beta \leftarrow \beta\times 2$

이렇게 갱신된 $\beta$는 다음 policy update에서 다시 사용된다. 이 방식에서는 가끔 실제 KL이 목표값과 꽤 크게 달라지는 업데이트가 나타나기도 하지만, 그런 경우는 드물고 $\beta$가 빠르게 보정된다고 저자들은 설명한다. 1.5와 2라는 숫자는 heuristic하게 정해졌지만 알고리즘은 이 값들에 아주 민감하지 않다. 또한 초기 $\beta$ 역시 하이퍼파라미터이지만, 실제로는 알고리즘이 빠르게 스스로 조정하기 때문에 큰 의미를 갖지 않는다.

## 5. Algorithm

앞선 절에서 소개한 surrogate loss들은 전형적인 policy gradient 구현을 아주 조금만 수정해도 계산하고 미분할 수 있다. 자동미분을 사용하는 구현이라면, 기존의 $L^{PG}$ 대신 $L^{CLIP}$이나 $L^{KLPEN}$을 loss로 정의하고, 그 objective에 대해 여러 번의 stochastic gradient ascent를 수행하면 된다.

분산이 작은 advantage estimator를 만들기 위해서는 보통 학습된 상태가치함수 $V(s)$가 필요하다. 예를 들어 generalized advantage estimation이나 finite-horizon estimator가 여기에 해당한다. 만약 policy와 value function이 파라미터를 공유하는 신경망 구조를 사용한다면, policy surrogate뿐 아니라 value function의 오차도 함께 줄이는 결합 objective가 필요하다. 여기에 exploration을 유지하기 위한 entropy bonus를 더하면 다음과 같은 형태를 얻는다.

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
\right]
\tag{9}
$$

여기서 $c_1$과 $c_2$는 계수이고, $S$는 entropy bonus를 뜻하며, $L_t^{VF}$는 다음의 제곱오차 형태를 가진다.

$$
L_t^{VF}(\theta)
=
\left(
V_\theta(s_t) - V_t^{\mathrm{targ}}
\right)^2
$$

정책을 매우 긴 episode 전체가 아니라 길이 $T$인 짧은 trajectory segment로 잘라 업데이트하는 구현 방식은 [Mni+16] 이후 널리 사용되었고, recurrent neural network와도 잘 맞는다. 이 방식에서는 advantage estimator가 timestep $T$ 이후를 보지 않아야 한다. [Mni+16]에서 사용한 estimator는 다음과 같다.

$$
\hat A_t
=
- V(s_t)
+ r_t
+ \gamma r_{t+1}
+ \cdots
+ \gamma^{T-t-1} r_{T-1}
+ \gamma^{T-t} V(s_T)
\tag{10}
$$

여기서 $t$는 길이 $T$인 trajectory segment 내부의 시간 인덱스다. 이 식은 segment의 끝에서 value bootstrap을 사용하는 finite-horizon advantage라고 볼 수 있다. 저자들은 이 선택을 일반화하여 truncated generalized advantage estimation도 사용할 수 있다고 말한다. 그 형태는 다음과 같고, $\lambda = 1$이면 식 (10)으로 돌아간다.

$$
\hat A_t
=
\delta_t
+
(\gamma\lambda)\delta_{t+1}
+
\cdots
+
(\gamma\lambda)^{T-t-1}\delta_{T-1}
\tag{11}
$$

$$
\delta_t
=
r_t + \gamma V(s_{t+1}) - V(s_t)
\tag{12}
$$

고정 길이 trajectory segment를 사용하는 PPO 알고리즘은 다음과 같이 정리된다.

### Algorithm 1. PPO, Actor-Critic Style

```text
for iteration = 1, 2, ... do
    for actor = 1, 2, ..., N do
        현재 policy π_{θ_old}를 사용해 환경에서 T timestep 동안 rollout을 수행한다.
        수집된 구간에 대해 advantage estimate Â_1, ..., Â_T를 계산한다.
    end for

    수집된 총 NT timestep 데이터에 대해 surrogate L을 θ에 관해 최적화한다.
    최적화는 K epochs 동안, minibatch size M ≤ NT로 수행한다.

    θ_old ← θ
end for
```

각 iteration에서 병렬 actor $N$개가 길이 $T$의 데이터를 모으고, 이 데이터 전체를 여러 epoch에 걸쳐 minibatch로 재사용한다는 점이 PPO의 핵심이다. 기존의 vanilla policy gradient에서는 같은 batch를 여러 번 쓰면 정책이 쉽게 망가졌지만, PPO에서는 Section 3의 clipped surrogate가 그 재사용을 가능하게 해 준다.

## 6. Experiments

### 6.1 Comparison of Surrogate Objectives

첫 번째 실험에서는 서로 다른 surrogate objective가 하이퍼파라미터에 따라 어떤 성능을 보이는지 비교한다. 여기서 비교하는 대상은 PPO의 $L^{CLIP}$뿐 아니라 그에 대한 자연스러운 변형과 ablation이다. 저자들은 다음 세 종류의 objective를 놓고 비교한다.

- clipping도 penalty도 없는 경우:
  $$
  L_t(\theta) = r_t(\theta)\,\hat A_t
  $$
- clipping을 사용하는 경우:
  $$
  L_t(\theta)
  =
  \min\!\left(
  r_t(\theta)\,\hat A_t,\;
  \operatorname{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\,\hat A_t
  \right)
  $$
- KL penalty를 사용하는 경우:
  $$
  L_t(\theta)
  =
  r_t(\theta)\,\hat A_t
  -
  \beta\,KL[\pi_{\theta_{\mathrm{old}}}, \pi_\theta]
  $$

KL penalty에는 고정된 penalty coefficient $\beta$를 사용할 수도 있고, Section 4에서 설명한 adaptive coefficient를 사용할 수도 있다. 저자들은 log-space에서 clipping을 시도해 보기도 했지만, 성능은 더 좋아지지 않았다고 보고한다.

각 알고리즘 변형마다 하이퍼파라미터를 탐색해야 했기 때문에, 이 실험은 계산 비용이 비교적 낮은 benchmark에서 수행되었다. 사용한 환경은 MuJoCo 물리 엔진을 사용하는 OpenAI Gym의 7개 시뮬레이션 로봇 제어 과제이고,[^2] 각 과제마다 1 million timestep 동안 학습을 진행한다. Clipping의 하이퍼파라미터 $\epsilon$과 KL penalty 쪽 하이퍼파라미터 $(\beta, d_{\mathrm{targ}})$를 제외한 나머지 설정은 Appendix A의 Table 3에 정리되어 있다.

정책 표현에는 hidden layer 두 개, 각 64 units, tanh 비선형성을 갖는 fully-connected MLP를 사용한다. 출력은 Gaussian distribution의 평균이며, 표준편차는 가변적인 값으로 둔다. 이 실험에서는 policy와 value function이 파라미터를 공유하지 않기 때문에 coefficient $c_1$은 의미가 없고, entropy bonus도 사용하지 않는다.

모든 알고리즘은 7개 환경 각각에서 3개의 random seed로 실행되었다. 각 실행은 마지막 100 episode의 평균 총보상으로 평가한다. 그리고 환경별 점수를 random policy가 0, 가장 좋은 결과가 1이 되도록 shift와 scale을 적용한 뒤, 총 21개의 실행 결과를 평균하여 하나의 scalar score를 얻는다. 이 설정에서 clipping이나 penalty가 전혀 없는 경우 점수가 음수가 되는데, 이는 HalfCheetah 환경에서 초기 random policy보다도 더 나쁜 매우 낮은 성능이 나타났기 때문이다.

<figure>
  <img src="/assets/images/Proximal_Policy_Optimization_Algorithms/03_Table_1.png" alt="Table 1. Normalized scores for surrogate objective variants on the MuJoCo benchmark." />
  <figcaption><strong>Table 1.</strong> Normalized scores for surrogate objective variants on the MuJoCo benchmark.</figcaption>
</figure>

Table 1의 결과를 보면 clipping을 사용하는 설정이 전반적으로 가장 좋고, 그중에서도 $\epsilon = 0.2$가 평균 정규화 점수 0.82로 가장 높은 값을 보인다. 반대로 clipping과 penalty가 없는 설정은 -0.39로 가장 낮다. Adaptive KL penalty와 fixed KL penalty도 어느 정도의 성능은 내지만, 전체적으로는 clipped surrogate objective보다 뒤처진다. 이 절의 결론은 이후 PPO 비교 실험에서 clipped surrogate objective를 대표 버전으로 사용해도 된다는 경험적 근거를 제공한다.

### 6.2 Comparison to Other Algorithms in the Continuous Domain

다음으로 저자들은 Section 3의 clipped surrogate objective를 사용하는 PPO를, 연속 제어 문제에서 효과적인 것으로 알려진 다른 알고리즘들과 비교한다. 비교 대상은 TRPO, cross-entropy method(CEM), adaptive stepsize를 사용하는 vanilla policy gradient,[^3] A2C, 그리고 trust region을 결합한 A2C이다.

A2C는 advantage actor critic의 약자이며, 저자들은 비동기식 A3C보다 동기식 A2C가 같은 수준 혹은 그 이상의 성능을 보인다고 설명한다. PPO는 앞 절과 동일한 하이퍼파라미터를 사용하고, clipping parameter는 $\epsilon = 0.2$로 둔다.

Figure 3은 HalfCheetah-v1, Hopper-v1, InvertedDoublePendulum-v1, InvertedPendulum-v1, Reacher-v1, Swimmer-v1, Walker2d-v1에서의 학습 곡선을 보여 준다. 논문의 요지는 PPO가 거의 모든 continuous control environment에서 기존 방법들보다 우수한 성능을 보인다는 것이다.

<figure>
  <img src="/assets/images/Proximal_Policy_Optimization_Algorithms/04_Figure_3.png" alt="Figure 3. Continuous control learning curves comparing PPO with other methods." />
  <figcaption><strong>Figure 3.</strong> Continuous control learning curves comparing PPO with other methods.</figcaption>
</figure>

### 6.3 Showcase in the Continuous Domain: Humanoid Running and Steering

이 절은 PPO가 단지 표준 MuJoCo benchmark에서만 잘 동작하는 것이 아니라, 고차원 연속 제어에서도 의미 있는 행동을 학습할 수 있음을 보여 주는 showcase다. 저자들은 3D humanoid를 대상으로 한 여러 과제를 사용한다. 여기서 로봇은 앞으로 달릴 뿐 아니라, 목표 방향으로 조향하고, 넘어졌다가 다시 일어나며, 경우에 따라서는 큐브에 맞으면서도 행동을 지속해야 한다.

실험에 사용한 세 과제는 다음과 같다.

1. **RoboschoolHumanoid**: 전방 보행만 수행하는 과제
2. **RoboschoolHumanoidFlagrun**: 200 timestep마다 혹은 목표에 도달할 때마다 목표 위치가 바뀌는 과제
3. **RoboschoolHumanoidFlagrunHarder**: 큐브에 맞으면서도 다시 일어나 목표를 향해 움직여야 하는 과제

학습 곡선은 Figure 4에, 학습된 정책의 정지 프레임은 Figure 5에 제시되어 있다. 사용한 하이퍼파라미터는 Table 4에 정리되어 있으며, 저자들은 동시에 진행되던 다른 연구에서도 Section 4의 adaptive KL 변형 PPO가 3D 로봇의 locomotion policy를 학습하는 데 사용되었다고 덧붙인다.

<figure>
  <img src="/assets/images/Proximal_Policy_Optimization_Algorithms/05_Figure_4.png" alt="Figure 4. Learning curves for the Roboschool humanoid tasks." />
  <figcaption><strong>Figure 4.</strong> Learning curves for the Roboschool humanoid tasks.</figcaption>
</figure>

<figure>
  <img src="/assets/images/Proximal_Policy_Optimization_Algorithms/06_Figure_5.png" alt="Figure 5. Snapshots of the learned humanoid behaviors in Roboschool." />
  <figcaption><strong>Figure 5.</strong> Snapshots of the learned humanoid behaviors in Roboschool.</figcaption>
</figure>

### 6.4 Comparison to Other Algorithms on the Atari Domain

저자들은 PPO를 Atari domain에서도 평가한다. 실험은 Arcade Learning Environment benchmark에서 수행되며, well-tuned A2C와 ACER와 비교한다. 세 알고리즘은 모두 [Mni+16]에서 사용한 것과 같은 policy network architecture를 사용한다. PPO 하이퍼파라미터는 Table 5에 정리되어 있고, 나머지 두 알고리즘은 이 benchmark에서 최대 성능을 내도록 별도로 조정된 하이퍼파라미터를 사용한다.

49개 게임 전체에 대한 결과 표와 학습 곡선은 Appendix B에 수록되어 있다. 본문에서는 두 가지 scoring metric을 사용한다. 첫 번째는 학습 전 구간에 걸친 episode 평균 보상으로, 빠른 학습을 선호하는 지표다. 두 번째는 학습 마지막 100 episode의 평균 보상으로, 최종 성능을 더 중시하는 지표다. Table 2는 각 지표에서 어떤 알고리즘이 몇 개의 게임을 승리했는지를 세 번의 trial 평균 기준으로 정리한다.

<figure>
  <img src="/assets/images/Proximal_Policy_Optimization_Algorithms/07_Table_2.png" alt="Table 2. Number of Atari games won by each algorithm under two scoring metrics." />
  <figcaption><strong>Table 2.</strong> Number of Atari games won by each algorithm under two scoring metrics.</figcaption>
</figure>

전 구간 평균 보상 기준에서는 PPO가 30개 게임에서 승리하고, ACER가 18개, A2C가 1개 게임에서 승리한다. 마지막 100 episode 평균 보상 기준에서는 ACER가 28개 게임에서, PPO가 19개 게임에서, A2C가 1개 게임에서 승리하며, 1개 게임은 동률이다. 즉 PPO는 학습 속도 측면에서 특히 강하고, 최종 성능에서는 ACER와 경쟁력 있는 결과를 낸다.

## 7. Conclusion

이 논문은 PPO라는 policy optimization 방법군을 제안했다. PPO는 각 policy update에서 여러 epoch의 stochastic gradient ascent를 수행하면서도 trust-region 계열 방법이 주는 안정성과 신뢰성을 상당 부분 유지한다.

또한 PPO는 vanilla policy gradient 구현을 기준으로 몇 줄의 코드만 바꾸면 될 만큼 단순하며, policy와 value function을 하나의 구조로 묶는 더 일반적인 설정에도 쉽게 적용할 수 있다. 논문의 전체 실험 결과는 PPO가 구현 단순성, 안정성, 그리고 실제 성능의 균형 측면에서 매우 강력한 선택지임을 보여 준다.

## 8. Acknowledgements

저자들은 Rocky Duan, Peter Chen, 그리고 OpenAI의 다른 동료들에게 유익한 의견을 준 것에 대해 감사를 표한다.

[^1]: DQN은 Arcade Learning Environment처럼 이산 행동 공간을 가진 게임에서는 잘 작동하지만, OpenAI Gym의 연속 제어 benchmark에서 좋은 성능을 보인다고는 아직 입증되지 않았다는 것이 논문의 설명이다.
[^2]: HalfCheetah, Hopper, InvertedDoublePendulum, InvertedPendulum, Reacher, Swimmer, Walker2d이며 모두 `-v1` 버전이다.
[^3]: 각 batch 뒤에 원래 policy와 업데이트된 policy 사이의 KL divergence를 기준으로 Adam step size를 조정한다. 규칙은 Section 4와 유사하다.

## References

- **[Bel+15]** M. Bellemare, Y. Naddaf, J. Veness, and M. Bowling. *The Arcade Learning Environment: An Evaluation Platform for General Agents.* Twenty-Fourth International Joint Conference on Artificial Intelligence, 2015.
- **[Bro+16]** G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba. *OpenAI Gym.* arXiv preprint arXiv:1606.01540, 2016.
- **[Dua+16]** Y. Duan, X. Chen, R. Houthooft, J. Schulman, and P. Abbeel. *Benchmarking Deep Reinforcement Learning for Continuous Control.* arXiv preprint arXiv:1604.06778, 2016.
- **[Hee+17]** N. Heess, S. Sriram, J. Lemmon, J. Merel, G. Wayne, Y. Tassa, T. Erez, Z. Wang, A. Eslami, M. Riedmiller, et al. *Emergence of Locomotion Behaviours in Rich Environments.* arXiv preprint arXiv:1707.02286, 2017.
- **[KL02]** S. Kakade and J. Langford. *Approximately Optimal Approximate Reinforcement Learning.* ICML, Vol. 2, 2002, pp. 267–274.
- **[KB14]** D. Kingma and J. Ba. *Adam: A Method for Stochastic Optimization.* arXiv preprint arXiv:1412.6980, 2014.
- **[Mni+15]** V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al. *Human-Level Control Through Deep Reinforcement Learning.* Nature 518(7540), 2015, pp. 529–533.
- **[Mni+16]** V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. P. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu. *Asynchronous Methods for Deep Reinforcement Learning.* arXiv preprint arXiv:1602.01783, 2016.
- **[Sch+15a]** J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv preprint arXiv:1506.02438, 2015.
- **[Sch+15b]** J. Schulman, S. Levine, P. Moritz, M. I. Jordan, and P. Abbeel. *Trust Region Policy Optimization.* CoRR, abs/1502.05477, 2015.
- **[SL06]** I. Szita and A. Lőrincz. *Learning Tetris Using the Noisy Cross-Entropy Method.* Neural Computation 18(12), 2006, pp. 2936–2941.
- **[TET12]** E. Todorov, T. Erez, and Y. Tassa. *MuJoCo: A Physics Engine for Model-Based Control.* Intelligent Robots and Systems (IROS), 2012 IEEE/RSJ International Conference, 2012, pp. 5026–5033.
- **[Wan+16]** Z. Wang, V. Bapst, N. Heess, V. Mnih, R. Munos, K. Kavukcuoglu, and N. de Freitas. *Sample Efficient Actor-Critic with Experience Replay.* arXiv preprint arXiv:1611.01224, 2016.
- **[Wil92]** R. J. Williams. *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.* Machine Learning 8(3–4), 1992, pp. 229–256.

## A. Hyperparameters

MuJoCo 1 million timestep benchmark에 사용한 PPO 하이퍼파라미터는 Table 3에 정리되어 있다.

<figure>
  <img src="/assets/images/Proximal_Policy_Optimization_Algorithms/08_Table_3.png" alt="Table 3. PPO hyperparameters for the MuJoCo 1M timestep benchmark." />
  <figcaption><strong>Table 3.</strong> PPO hyperparameters for the MuJoCo 1M timestep benchmark.</figcaption>
</figure>

Roboschool humanoid 실험에서 사용한 PPO 하이퍼파라미터는 Table 4에 정리되어 있다. 이 실험에서는 Adam step size가 목표 KL divergence 값에 맞춰 조정된다.

<figure>
  <img src="/assets/images/Proximal_Policy_Optimization_Algorithms/09_Table_4.png" alt="Table 4. PPO hyperparameters for the Roboschool humanoid experiments." />
  <figcaption><strong>Table 4.</strong> PPO hyperparameters for the Roboschool humanoid experiments.</figcaption>
</figure>

Atari 실험에서 사용한 PPO 하이퍼파라미터는 Table 5에 정리되어 있다. 여기서 $\alpha$는 학습이 진행되는 동안 1에서 0으로 선형 감소하며, 이에 따라 Adam step size와 clipping parameter도 함께 annealing된다.

<figure>
  <img src="/assets/images/Proximal_Policy_Optimization_Algorithms/10_Table_5.png" alt="Table 5. PPO hyperparameters for the Atari experiments." />
  <figcaption><strong>Table 5.</strong> PPO hyperparameters for the Atari experiments.</figcaption>
</figure>

## B. Performance on More Atari Games

Appendix B는 49개 Atari 게임 전체에 대한 학습 곡선과 평균 성능을 모아 놓은 부록이다. Figure 6은 각 게임에서 세 개의 random seed로 실행한 학습 곡선을 보여 주고, Table 6은 mean final performance를 정리한다.

<figure>
  <img src="/assets/images/Proximal_Policy_Optimization_Algorithms/11_Figure_6.png" alt="Figure 6. Atari learning curves for PPO, A2C, and ACER across 49 games." />
  <figcaption><strong>Figure 6.</strong> Atari learning curves for PPO, A2C, and ACER across 49 games.</figcaption>
</figure>

Table 6은 40M game frames, 즉 10M timesteps 이후의 마지막 100 episode 평균 점수를 게임별로 제시한다. 본문 Table 2가 승리한 게임 수를 요약한 것이라면, Table 6은 어떤 게임에서 어느 알고리즘이 강했는지를 개별적으로 확인하게 해 주는 표다.

<figure>
  <img src="/assets/images/Proximal_Policy_Optimization_Algorithms/12_Table_6.png" alt="Table 6. Mean final performance on each Atari game after training." />
  <figcaption><strong>Table 6.</strong> Mean final performance on each Atari game after training.</figcaption>
</figure>

## 추가 설명

### 1. 이 논문이 정확히 해결하려는 문제

이 논문은 단순히 정책을 더 잘 학습하는 새 알고리즘을 내놓는 데서 끝나지 않는다. 저자들이 직접 겨냥하는 문제는 훨씬 구체적이다. On-policy policy gradient 방법은 현재 policy로 모은 데이터를 사용한다는 점에서 안정적이지만, 같은 데이터를 여러 번 재사용하기가 어렵다. 같은 batch로 update를 너무 많이 하면 policy가 data-collecting policy에서 멀어지고, 그러면 그 데이터는 더 이상 현재 policy를 잘 대표하지 못한다. 반대로 update를 아주 조금만 하면 안정적이지만 sample efficiency가 낮아진다. PPO는 바로 이 두 요구, 즉 같은 데이터를 여러 번 쓰고 싶다와 policy가 한 번에 너무 멀리 가면 안 된다는 조건을 동시에 만족시키려는 시도다.

TRPO는 이 문제를 KL constraint로 해결한다. 즉 old policy와 new policy의 거리를 제한해서 너무 큰 update를 막는다. 하지만 TRPO는 second-order 근사, conjugate gradient, 별도의 constraint 처리 같은 구현 복잡성이 있다. PPO의 목표는 TRPO의 철학은 유지하면서도, 훨씬 단순한 first-order update만으로 비슷한 안정성을 얻는 것이다.

### 2. 식 (1)과 식 (2): policy gradient의 기본 형태

식 (1)은 REINFORCE 계열에서 가장 기본이 되는 score-function gradient estimator다.

$$
\hat g
=
\hat{\mathbb E}_t
\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\hat A_t
\right]
$$

이 식은 해석이 직관적이다. 어떤 시점의 action이 예상보다 좋았다면, 즉 $\hat A_t > 0$이라면 그 action의 로그확률을 증가시키는 방향으로 update가 일어난다. 반대로 어떤 action이 예상보다 나빴다면, 즉 $\hat A_t < 0$이라면 그 action의 로그확률을 낮추는 방향으로 update가 일어난다. 결국 policy gradient는 좋은 행동은 더 자주, 나쁜 행동은 덜 자주 선택하게 만드는 방향의 gradient다.

식 (2)는 이 gradient를 자동미분으로 얻기 위한 objective다.

$$
L^{PG}(\theta)
=
\hat{\mathbb E}_t
\left[
\log \pi_\theta(a_t \mid s_t)\,\hat A_t
\right]
$$

중요한 점은 이 objective 자체가 update의 크기를 통제하지 않는다는 것이다. 한 번의 small step에는 문제가 없지만, 같은 batch에 대해 여러 번 gradient ascent를 수행하면 $\log \pi_\theta(a_t \mid s_t)$를 과도하게 밀어 올리거나 밀어 내리게 된다. 그 결과 policy가 old policy에서 너무 멀리 이동하고, 이때는 gradient 방향 자체가 더 이상 믿을 만하지 않게 된다. 논문이 그냥 식 (2)를 여러 번 최적화하는 방식은 잘 정당화되지 않는다고 말하는 이유가 바로 여기에 있다.

### 3. 식 (3)부터 식 (5)까지: TRPO가 하려던 일

TRPO는 old policy에서 수집한 데이터를 new policy를 평가하는 데 사용하기 위해 probability ratio를 도입한다.

$$
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
$$

이 비율이 1보다 크면 new policy가 그 action을 old policy보다 더 선호한다는 뜻이고, 1보다 작으면 덜 선호한다는 뜻이다. 식 (3)은 이 비율로 advantage를 재가중해 new policy의 성능 변화를 근사한다. 하지만 이 surrogate objective만 믿고 크게 움직이면 문제가 생긴다. 그래서 식 (4)에서 평균 KL divergence를 제한해 update 크기를 직접 제어한다.

TRPO의 핵심 아이디어는 좋은 방향으로 가더라도 너무 멀리 한 번에 가지는 말자는 것이다. 식 (5)의 penalty version도 같은 철학을 담고 있다. 다만 penalty coefficient $\beta$ 하나만으로 모든 문제와 모든 학습 구간을 다 맞추기는 어렵다. 어떤 구간에서는 penalty가 너무 약해져 큰 update를 허용하고, 다른 구간에서는 너무 강해서 학습을 지나치게 느리게 만들 수 있다. PPO는 이 문제를 penalty coefficient를 정교하게 튜닝하는 대신, objective 자체를 변형하는 방향으로 해결한다.

### 4. 식 (6)과 식 (7): PPO-Clip의 핵심 메커니즘

PPO는 먼저 TRPO surrogate를 다음처럼 다시 쓴다.

$$
L^{CPI}(\theta)
=
\hat{\mathbb E}_t
\left[
r_t(\theta)\,\hat A_t
\right],
\qquad
r_t(\theta)
=
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
$$

그리고 여기에 clipping을 넣어 식 (7)을 만든다.

$$
L^{CLIP}(\theta)
=
\hat{\mathbb E}_t
\left[
\min\!\left(
r_t(\theta)\,\hat A_t,\;
\operatorname{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\,\hat A_t
\right)
\right]
$$

이 식이 왜 중요한지는 advantage의 부호에 따라 나누어 보면 분명해진다.

먼저 $\hat A_t > 0$인 경우를 보자. 이때는 그 action의 확률을 높이는 것이 유리하다. 따라서 $r_t(\theta)$가 1보다 커질수록 $r_t(\theta)\hat A_t$도 커진다. 하지만 PPO는 $r_t(\theta)$가 $1+\epsilon$을 넘어서면 더 이상의 추가 이득을 주지 않는다. 즉 좋은 행동의 확률을 올리는 것은 허용하되, 일정 범위를 넘는 과도한 증가에는 보상을 더 주지 않는다.

수식으로 쓰면 $\hat A_t > 0$일 때는 사실상 다음과 같은 꼴이 된다.

$$
\min\!\left(
r_t(\theta),\,1+\epsilon
\right)\hat A_t
$$

반대로 $\hat A_t < 0$이면 그 action은 나쁜 행동이므로 확률을 낮추는 방향이 유리하다. 이 경우에는 $r_t(\theta)$가 1보다 작아질수록 좋아지지만, PPO는 $r_t(\theta)$가 $1-\epsilon$보다 더 작아져도 그 추가적인 개선을 더 이상 보상하지 않는다. 즉 나쁜 행동의 확률을 낮추는 것도 어느 정도까지만 보상한다.

이 경우는 결과적으로 다음과 같은 해석이 가능하다.

$$
\max\!\left(
r_t(\theta),\,1-\epsilon
\right)\hat A_t
\qquad (\hat A_t < 0)
$$

부호가 음수이기 때문에 겉모양은 다르지만, 의미는 동일하다. 좋은 행동의 확률을 너무 많이 올리는 것도, 나쁜 행동의 확률을 너무 많이 내리는 것도 모두 막는다. PPO는 특정 방향의 update가 좋은 방향이라는 사실만으로 무제한 신뢰하지 않는다.

### 5. 왜 `min`을 쓰면 lower bound가 되는가

식 (7)에서 가장 많이 오해되는 부분이 바로 `min`이다. 직관만으로 보면 왜 굳이 작은 값을 택하지라는 의문이 생긴다. 하지만 이 연산이 바로 PPO의 보수적인 성격을 만든다.

원래 surrogate인 $r_t(\theta)\hat A_t$는 policy가 멀리 이동할수록 과도하게 낙관적인 값을 줄 수 있다. 반면 clipped term은 일정 범위를 넘는 비율 변화에 대해서는 더 이상의 추가 이득을 인정하지 않는다. 두 항 중 작은 값을 택하면, old data로 측정한 성능 개선을 항상 보수적으로 계산하게 된다. 그래서 논문은 이것을 pessimistic estimate, 또는 lower bound라고 부른다.

여기서 lower bound라는 표현은 성능을 엄밀하게 증명한 하한이라는 뜻으로 받아들이기보다, unclipped surrogate보다 더 보수적으로 행동하는 목적함수라는 의미로 이해하는 것이 좋다. Figure 2가 바로 그 점을 시각적으로 보여 준다.

### 6. Figure 1과 Figure 2를 어떻게 읽어야 하는가

Figure 1은 단 하나의 timestep 항만 떼어 놓고 본 그림이다. 왼쪽은 advantage가 양수일 때이고, 오른쪽은 advantage가 음수일 때다. 두 그림 모두 빨간 점이 시작점 $r=1$을 표시한다. 이 그림의 요점은 매우 단순하다. 최적화는 처음에는 일반 policy gradient처럼 움직이지만, $r$가 clipping 경계를 넘는 순간부터 objective가 더 이상 개선되지 않는다. 즉 PPO는 update 방향은 허용하되 update 크기는 직접적으로 제한한다.

Figure 2는 실제 Hopper-v1의 첫 번째 PPO update를 따라가면서, 여러 objective가 어떻게 변하는지 보여 준다. 이 그림에서 unclipped surrogate $L^{CPI}$는 계속 올라가지만, $L^{CLIP}$은 어느 지점에서 최고점을 찍고 다시 내려간다. 그리고 그 최고점은 old policy와 new policy 사이의 KL divergence가 약 0.02 정도 되는 지점과 대응한다. 이는 그 이상 가면 surrogate가 오히려 신뢰할 수 없어진다는 PPO의 철학을 잘 보여 준다. TRPO가 constraint로 하던 일을 PPO는 objective의 모양 자체로 흉내 내고 있다고 볼 수 있다.

### 7. 식 (8): adaptive KL penalty는 왜 baseline으로 남았는가

Adaptive KL penalty 버전은 다음과 같은 생각에서 출발한다. KL이 목표값보다 작으면 update가 너무 보수적이니 penalty를 약하게 하고, KL이 목표값보다 크면 update가 너무 공격적이니 penalty를 강하게 한다. 이 방식은 직관적으로 타당하며 실제 구현도 어렵지 않다.

하지만 논문 실험에서는 clipping이 더 잘 됐다. 이유를 직관적으로 말하면, KL penalty는 여전히 평균 KL 하나에 의존하는 방식이고, coefficient $\beta$가 순간순간 얼마나 잘 맞느냐에 따라 학습 성능이 달라진다. 반면 clipping은 timestep별 objective 수준에서 직접 과도한 ratio 변화에 대한 보상을 없애 버린다. 즉 penalty는 update 이후에 거리를 조절하는 느낌이고, clipping은 애초에 objective가 그런 방향을 덜 선호하게 만드는 느낌이다. 논문이 최종적으로 PPO-Clip을 대표 버전으로 선택한 이유가 여기에 있다.

### 8. 식 (9): 실제 구현에서 PPO loss는 어떻게 쓰이는가

실전 강화학습 구현에서 policy만 따로 학습하고 value function은 별도로 두는 경우도 있지만, 하나의 신경망이 policy와 value function을 함께 출력하는 경우도 많다. 이런 구조에서는 policy loss만 가지고는 충분하지 않다. Value function도 같이 학습해야 하고, exploration도 유지해야 한다. 그래서 논문은 다음 objective를 사용한다.

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
\right]
$$

여기서 $L_t^{VF}$는 value loss이고, $S$는 entropy bonus다. 기호를 보면 value loss 앞에 마이너스가 붙어 있는데, 이는 전체 식을 maximize할 objective로 썼기 때문이다. 실제 코드에서는 보통 optimizer가 loss를 minimize하도록 구현되어 있으므로 부호를 뒤집어서 쓰는 경우가 많다.

Entropy bonus가 들어가는 이유도 중요하다. PPO는 policy가 너무 빨리 한 방향으로 수렴해 exploration이 줄어드는 것을 막고 싶다. 특히 discrete action이나 sparse reward 환경에서는 entropy bonus가 안정성에 큰 도움을 준다. Table 5에서 Atari 설정에 entropy coefficient가 따로 들어 있는 이유도 여기에 있다.

### 9. 식 (10)부터 식 (12)까지: finite-horizon estimator와 truncated GAE

식 (10)은 짧은 trajectory segment 위에서 계산하는 finite-horizon advantage다.

$$
\hat A_t
=
- V(s_t)
+ r_t
+ \gamma r_{t+1}
+ \cdots
+ \gamma^{T-t-1} r_{T-1}
+ \gamma^{T-t} V(s_T)
$$

이 식은 segment 끝까지의 discounted return에 마지막 상태의 bootstrap value를 더하고 현재 상태 value를 뺀 구조다. Episode 끝까지 다 보지 않아도 되기 때문에 길이가 짧은 rollout과 잘 맞는다. RNN이나 병렬 actor 구조에서 특히 유용한 이유도 여기에 있다.

여기서 식 (10)부터 식 (12)까지의 $r_t$는 시점 $t$의 reward를 뜻하며, Section 3에서 사용한 probability ratio $r_t(\theta)$와는 다른 기호라는 점을 문맥으로 구분해야 한다.

식 (11)과 식 (12)는 truncated GAE다.

$$
\hat A_t
=
\delta_t
+
(\gamma\lambda)\delta_{t+1}
+
\cdots
+
(\gamma\lambda)^{T-t-1}\delta_{T-1}
$$

$$
\delta_t
=
r_t + \gamma V(s_{t+1}) - V(s_t)
$$

여기서 $\delta_t$는 1-step TD error이고, GAE는 이 TD error들을 지수적으로 가중합한다. $\lambda$가 1에 가까우면 긴 horizon의 정보를 더 많이 반영해 bias는 줄고 variance는 커진다. $\lambda$가 작아지면 더 짧은 horizon 쪽으로 기울어 variance는 줄지만 bias가 커진다. Appendix A를 보면 논문은 대부분의 실험에서 $\lambda = 0.95$를 사용한다. 이는 bias와 variance 사이의 절충값으로 널리 쓰이는 설정이다.

### 10. Algorithm 1을 실제 학습 루프로 읽는 법

Algorithm 1은 아주 짧게 적혀 있지만 PPO의 실행 구조를 정확히 담고 있다.

첫 단계에서는 old policy $\pi_{\theta_{\mathrm{old}}}$로 병렬 actor $N$개가 각자 $T$ timestep의 rollout을 수집한다. 둘째 단계에서는 각 segment에 대해 advantage estimate를 계산한다. 셋째 단계에서는 총 $NT$개의 timestep을 하나의 on-policy batch로 모아 surrogate objective를 만든다. 넷째 단계에서는 그 batch를 minibatch로 쪼개어 $K$ epochs 동안 여러 번 optimization한다. 마지막으로 $\theta_{\mathrm{old}} \leftarrow \theta$로 바꾸고 다시 rollout을 수집한다.

핵심은 PPO가 on-policy라는 사실을 유지하면서도, 한 번 모은 batch를 여러 epoch 재사용한다는 데 있다. Data replay buffer를 오래 유지하는 off-policy 방법은 아니지만, vanilla policy gradient보다 훨씬 효율적으로 같은 데이터를 활용한다. PPO가 실전에서 간단한데 잘 되는 baseline으로 받아들여진 이유 중 하나가 바로 이 구조다.

### 11. Table 1이 말해 주는 것

Table 1은 이 논문의 가장 중요한 실험 중 하나다. 저자들은 surrogate objective를 여러 방식으로 바꾸어 놓고, 어떤 형태가 실제로 가장 잘 작동하는지 직접 비교한다.

결과는 매우 분명하다. No clipping or penalty는 평균 정규화 점수 -0.39로 가장 나쁘다. 같은 batch를 여러 번 최적화하되 update를 통제하지 않으면 오히려 학습이 무너질 수 있다는 뜻이다. Clipping은 $\epsilon = 0.1$에서 0.76, $\epsilon = 0.2$에서 0.82, $\epsilon = 0.3$에서 0.70을 기록한다. 너무 작은 $\epsilon$은 update를 지나치게 묶어 버리고, 너무 큰 $\epsilon$은 보호 장치가 약해진다. 이 실험에서는 0.2가 가장 균형이 좋다.

Adaptive KL penalty는 $d_{\mathrm{targ}}=0.003$에서 0.68, 0.01에서 0.74, 0.03에서 0.71이고, fixed KL penalty는 $\beta=0.3, 1, 3, 10$에서 각각 0.62, 0.71, 0.72, 0.69다. 즉 KL penalty 계열도 전혀 나쁜 방법은 아니지만, clipping만큼 안정적으로 최고 성능을 내지는 못한다. 이 표 하나만으로도 왜 이후 PPO의 대표 형태가 PPO-Penalty가 아니라 PPO-Clip이 되었는지 이해할 수 있다.

### 12. Figure 3: 연속 제어 비교 결과의 의미

Figure 3은 PPO를 여러 기존 알고리즘과 나란히 놓고 학습 곡선을 비교한 그림이다. 이 그림에서 중요한 것은 최종 점수뿐 아니라 학습 과정 전체다. PPO는 1 million timestep 동안 대부분의 환경에서 높은 점수를 안정적으로 달성한다. 특히 HalfCheetah와 Walker2d에서는 PPO의 우세가 뚜렷하고, InvertedDoublePendulum과 InvertedPendulum처럼 빠르게 포화되는 과제에서도 강한 성능을 보인다.

이 그림이 중요한 이유는 PPO가 단순한데도 잘 된다는 주장을 단순한 수사가 아니라 실제 benchmark 결과로 보여 주기 때문이다. CEM, TRPO, A2C, trust-region actor-critic, adaptive vanilla policy gradient 등 여러 대안을 함께 놓고 비교했을 때 PPO가 거의 모든 환경에서 경쟁력을 유지한다는 것은, PPO의 강점이 특정 문제에만 국한되지 않는다는 뜻이다.

### 13. Figure 4와 Figure 5: 고차원 humanoid 제어에서의 의미

Section 6.3의 의미는 MuJoCo standard benchmark를 하나 더 돌렸다는 데 있지 않다. 여기서는 3D humanoid가 달리고, 방향을 바꾸고, 넘어졌다가 다시 일어나고, 외부 방해까지 버텨야 한다. 즉 상태공간도 크고, 행동공간도 크며, 필요한 기술도 단순 주행을 넘어선다.

Figure 4의 학습 곡선은 PPO가 이런 고차원 제어에서도 장기간 학습을 견디며 성능을 끌어올릴 수 있음을 보여 준다. Figure 5의 정지 프레임은 정량적 곡선만으로는 드러나지 않는 행동의 질을 보여 준다. 처음 여섯 프레임에서는 로봇이 현재 목표를 향해 달리고, 그다음 목표 위치가 바뀌자 몸의 방향을 바꾸어 새로운 목표를 향해 달린다. 이 그림은 PPO가 단순히 reward를 높이는 것뿐 아니라, 목표 변화에 반응하는 정책을 실제로 학습했다는 질적 증거다.

### 14. Table 2와 Appendix B를 함께 읽는 방법

Atari 결과는 두 단계로 읽는 것이 좋다. 먼저 Table 2는 요약표다. 전 구간 평균 보상이라는 지표에서는 PPO가 30개 게임에서 승리하고 ACER가 18개, A2C가 1개 게임에서 승리한다. 이 지표는 학습 속도를 더 중요하게 보기 때문에, PPO가 빠르게 좋은 정책에 도달하는 경향이 강하다는 뜻이다.

반면 마지막 100 episode 평균 보상이라는 지표에서는 ACER가 28개 게임에서 승리하고 PPO가 19개, A2C가 1개 게임에서 승리하며 1개 게임은 동률이다. 이 지표는 최종 성능을 더 중시하므로, ACER가 일부 게임에서는 더 높은 asymptotic performance를 낸다고 해석할 수 있다. 그래서 논문 Introduction의 표현도 정확하다. PPO는 A2C보다 sample complexity 면에서 확실히 낫고, ACER와는 비슷한 수준의 성능을 보이지만 훨씬 단순하다.

Appendix B의 Table 6은 이 요약의 내부를 들여다보게 해 준다. PPO는 Assault, BattleZone, Enduro, Freeway, Gravitar, Jamesbond, Kangaroo, MontezumaRevenge, Robotank, Tennis, TimePilot, WizardOfWor, Zaxxon 같은 게임에서 강한 최종 성능을 보인다. 반면 ACER는 Asterix, Boxing, Breakout, Centipede, ChopperCommand, DemonAttack, Gopher, MsPacman, Qbert, RoadRunner, SpaceInvaders, StarGunner, UpNDown, VideoPinball 등에서 강하다. Pong은 PPO와 ACER가 같은 20.7로 사실상 동률이고, Venture는 세 알고리즘 모두 0.0으로 끝난다. 이런 세부 결과를 보면 PPO가 모든 게임을 압도하는 것은 아니지만, 매우 넓은 범위의 게임에서 고르게 강하다는 점이 드러난다.

### 15. Appendix A의 하이퍼파라미터가 의미하는 바

Table 3을 보면 MuJoCo benchmark에서는 horizon 2048, Adam stepsize $3\times 10^{-4}$, 10 epochs, minibatch size 64, discount 0.99, GAE parameter 0.95를 사용한다. 이 조합은 비교적 긴 rollout으로 on-policy batch를 만들고, 그 batch를 여러 번 반복 학습하는 전형적인 PPO 설정이다.

Table 4의 Roboschool 설정은 MuJoCo와 다르다. Horizon은 512로 줄어들지만 minibatch size는 4096으로 매우 크고, epoch도 15회로 늘어난다. Actor 수는 locomotion에서 32, flagrun에서 128이고, action distribution의 log standard deviation은 -0.7에서 -1.6으로 선형적으로 annealing된다. 또한 Adam stepsize는 고정값이 아니라 target KL에 맞추어 조정된다. 이는 고차원 humanoid task에서 exploration과 안정성을 동시에 관리하려는 설계로 읽을 수 있다.

Table 5의 Atari 설정도 domain 특성을 반영한다. Horizon은 128, epochs는 3, number of actors는 8이고, Adam stepsize는 $2.5\times 10^{-4}\times \alpha$, clipping parameter는 $0.1\times \alpha$다. 여기서 $\alpha$는 학습이 진행되면서 1에서 0으로 선형 감소한다. 즉 학습 초반에는 큰 step과 상대적으로 넓은 clip range로 빠르게 배우고, 후반으로 갈수록 더 보수적인 update를 하도록 설계된 것이다. Value-function coefficient $c_1=1$과 entropy coefficient $c_2=0.01$이 명시되어 있다는 점도 Atari 실험이 분명한 actor-critic 구조 위에서 돌아간다는 것을 보여 준다.

### 16. PPO가 실제로 강한 이유를 한 문장으로 요약하면

PPO의 강점은 정책을 너무 멀리 움직이면 안 된다는 TRPO의 핵심 직관을, second-order constraint solver 없이 objective의 모양만으로 구현했다는 데 있다. 그래서 구현은 단순하고, batch를 여러 번 재사용할 수 있어 효율적이며, update가 쉽게 폭주하지 않는다.

이 논문이 이후 강화학습에서 큰 영향력을 갖게 된 것도 바로 이 지점 때문이다. PPO는 새로운 이론적 프레임워크를 복잡하게 도입한 논문이라기보다, 기존에 알고 있던 문제를 매우 실용적인 방식으로 정리한 논문이다. 식 (7)의 clipping이라는 한 줄짜리 아이디어가 실제 학습 안정성과 구현 편의성 사이의 균형을 크게 바꾸어 놓았다는 점이 이 논문의 핵심 성과다.

### 17. 이 논문을 읽을 때 끝까지 붙잡아야 하는 문장

이 논문의 본질은 같은 on-policy 데이터를 여러 번 써도 망가지지 않게 만들자라는 한 문장으로 압축할 수 있다. Introduction의 문제의식, Section 3의 clipping, Section 5의 반복적 minibatch optimization, Section 6.1의 objective comparison, 그리고 이후의 benchmark 성능까지 모두 이 문장을 중심으로 연결된다. PPO를 이해했다는 것은 식 (7)을 외웠다는 뜻이 아니라, 왜 식 (7)이 여러 epoch의 on-policy optimization을 가능하게 만드는지까지 이해했다는 뜻이다.